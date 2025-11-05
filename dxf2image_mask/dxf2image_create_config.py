#!/usr/bin/env python3
"""
DXF → Image mask with **6-point statistical homography**

Workflow
- Provide **exact 6 DXF coordinates** (clockwise NOT required, just consistent order).
- Click the **same 6 points** on the photo in the **same order**.
- We solve a homography with least squares from all 6 pairs, compute residual stats,
  draw per-point error vectors, then warp the DXF polygon and export a black/white mask.

What you get
- Console stats: per-point error (px), RMSE, median, 95th percentile, max.
- A diagnostics PNG with error vectors and the warped polygon overlay (optional).
- The mask PNG (black background, white filled polygon).
- **NEW:** A mask **overlay** on the original image for quick visual control.

Usage
    # Let dialogs pick image, dxf, out, and CSV with 6 coords:
    python dxf2mask.py

    # Or specify the CSV directly (6 rows x,y; header allowed; comma/semicolon):
    python dxf2mask.py --src-csv six_points.csv --save-overlay overlay.png --save-diagnostics diag.png

Options
    --layer NAME            Only consider entities on this DXF layer
    --tolerance 0.001       Endpoint snap tolerance for rebuilding closed LINE chains
    --src-csv path.csv      CSV/TXT with **6** rows of x,y in DXF units (header allowed)
    --src-corners "x1,y1;...;x6,y6"  Inline 6 pairs
    --save-diagnostics path Optional PNG path for the error visualization
    --ransac                Use RANSAC fit first (helps if 1 point is bad)
    --ransac-thresh 4.0     RANSAC reprojection threshold in pixels
    --save-config path      Optional JSON/TXT config export with src/dst points

Dependencies: ezdxf, opencv-python, numpy, tkinter (file pickers)
"""

import argparse
import cv2
import numpy as np
import ezdxf
import json
from typing import List, Tuple, Optional

# For interactive file selection
try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_OK = True
except Exception:
    _TK_OK = False

# ------------------------
# Helpers
# ------------------------

def _clockwise_order_quad(points: np.ndarray) -> np.ndarray:
    # Kept for polygon bbox use in other helpers if needed; not used for the 6-point fit
    c = np.mean(points, axis=0)
    ang = np.arctan2(points[:,1]-c[1], points[:,0]-c[0])
    ordered = points[np.argsort(ang)]
    x, y = ordered[:,0], ordered[:,1]
    s = 0.5 * np.sum(x*np.roll(y, -1) - y*np.roll(x, -1))
    if s < 0:
        ordered = ordered[::-1]
    return ordered


def _ensure_closed(pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if len(pts) < 3:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        pts = np.vstack([pts, pts[0]])
    return pts


def _area_abs(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _largest_closed_poly_from_msp(msp, layer: Optional[str] = None, line_snap_tol: float = 1e-3) -> np.ndarray:
    polys: List[np.ndarray] = []

    def on_layer(e) -> bool:
        if layer is None:
            return True
        try:
            return (e.dxf.layer == layer)
        except Exception:
            return True

    # 1) LWPOLYLINE / POLYLINE
    for e in msp.query("LWPOLYLINE POLYLINE"):
        if not on_layer(e):
            continue
        try:
            if e.dxftype() == "LWPOLYLINE":
                pts = np.array([(p[0], p[1]) for p in e.get_points()], dtype=np.float32)
                if e.closed or getattr(e, "is_closed", False) or (len(pts) >= 3 and np.allclose(pts[0], pts[-1])):
                    polys.append(_ensure_closed(pts))
            else:
                pts = np.array([(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()], dtype=np.float32)
                if e.is_closed or getattr(e, "closed", False) or (len(pts) >= 3 and np.allclose(pts[0], pts[-1])):
                    polys.append(_ensure_closed(pts))
        except Exception:
            continue

    # 2) HATCH
    for h in msp.query("HATCH"):
        if not on_layer(h):
            continue
        try:
            for path in h.paths:
                pts = None
                try:
                    poly_path = path.to_polyline_path(0.5)
                    pts = np.array([(v.x, v.y) for v in poly_path.vertices], dtype=np.float32)
                except Exception:
                    pts_list: List[Tuple[float, float]] = []
                    for ed in getattr(path, "edges", []):
                        et = getattr(ed, 'EDGE_TYPE', '')
                        if et == "LineEdge":
                            pts_list.append((ed.start.x, ed.start.y))
                        elif et in ("ArcEdge", "EllipseEdge"):
                            steps = 64
                            for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                                try:
                                    p = ed.point_at(t)
                                    pts_list.append((p.x, p.y))
                                except Exception:
                                    pass
                    if len(pts_list) >= 3:
                        pts_list.append(pts_list[0])
                        pts = np.array(pts_list, dtype=np.float32)
                if pts is not None and len(pts) >= 3:
                    polys.append(_ensure_closed(pts))
        except Exception:
            continue

    # 3) SPLINE
    for s in msp.query("SPLINE"):
        if not on_layer(s):
            continue
        try:
            approx = np.array(s.approximate(200), dtype=np.float32)
            if len(approx) >= 3:
                polys.append(_ensure_closed(approx))
        except Exception:
            continue

    # 4) CIRCLE / ELLIPSE
    for c in msp.query("CIRCLE ELLIPSE"):
        if not on_layer(c):
            continue
        try:
            steps = 180
            if c.dxftype() == "CIRCLE":
                center = np.array([c.dxf.center.x, c.dxf.center.y], dtype=np.float32)
                r = float(c.dxf.radius)
                theta = np.linspace(0, 2*np.pi, steps, endpoint=False)
                pts = np.stack([center[0] + r*np.cos(theta), center[1] + r*np.sin(theta)], axis=1).astype(np.float32)
            else:
                center = np.array([c.dxf.center.x, c.dxf.center.y], dtype=np.float32)
                major = np.array([c.dxf.major_axis.x, c.dxf.major_axis.y], dtype=np.float32)
                ratio = float(c.dxf.radius_ratio)
                theta = np.linspace(0, 2*np.pi, steps, endpoint=False)
                minor = np.array([-major[1], major[0]], dtype=np.float32)
                pts = np.array([center + np.cos(t)*major + np.sin(t)*ratio*minor for t in theta], dtype=np.float32)
            polys.append(_ensure_closed(pts))
        except Exception:
            continue

    # 5) LINE chains
    lines = []
    for ln in msp.query("LINE"):
        if not on_layer(ln):
            continue
        try:
            p0 = np.array([ln.dxf.start.x, ln.dxf.start.y], dtype=np.float32)
            p1 = np.array([ln.dxf.end.x, ln.dxf.end.y], dtype=np.float32)
            lines.append((p0, p1))
        except Exception:
            pass
    if lines:
        tol = float(line_snap_tol)
        nodes: List[np.ndarray] = []
        def find_or_add(pt: np.ndarray) -> int:
            for i, n in enumerate(nodes):
                if np.linalg.norm(pt - n) <= tol:
                    return i
            nodes.append(pt)
            return len(nodes)-1
        edges = []
        for a, b in lines:
            ia, ib = find_or_add(a), find_or_add(b)
            edges.append((ia, ib))
        if nodes:
            adj = {i: [] for i in range(len(nodes))}
            for ia, ib in edges:
                adj[ia].append(ib)
                adj[ib].append(ia)
            start = next((i for i, nbrs in adj.items() if len(nbrs) == 2), 0)
            path = [start]
            prev = None
            cur = start
            for _ in range(len(edges) + 1):
                nbrs = adj[cur]
                nxts = [n for n in nbrs if n != prev] if prev is not None else nbrs
                if not nxts:
                    break
                nxt = nxts[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == start and len(path) > 3:
                    break
            if len(path) > 3 and path[-1] == start:
                pts = np.array([nodes[i] for i in path], dtype=np.float32)
                polys.append(_ensure_closed(pts))

    if not polys:
        raise ValueError("No closed polygonal geometry found in DXF (polylines/hatches/splines/circles/lines).")

    polys.sort(key=_area_abs, reverse=True)
    return polys[0]


class ClickCollector:
    def __init__(self, image, needed: int, window_name: str = None, max_size: Tuple[int, int] = (1400, 900)):
        self.img_orig = image
        self.h0, self.w0 = image.shape[:2]
        self.needed = needed
        self.points: List[Tuple[int, int]] = []  # stored in ORIGINAL pixel coords
        self.window = window_name or f"Click {needed} destination points (press any key when done)"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # --- Aspect-ratio preserving scale to fit window ---
        mw, mh = max_size
        s = min(mw / self.w0, mh / self.h0)
        s = min(s, 1.0)  # don't upscale by default
        self.scale = float(s)
        self.disp = cv2.resize(self.img_orig, (int(self.w0 * self.scale), int(self.h0 * self.scale)), interpolation=cv2.INTER_AREA)
        self.overlay = self.disp.copy()

        # Zoom settings
        self.zoom_on = True
        self.zoom_factor = 3.0
        self.zoom_box = 60  # half-size of crop in original pixels
        self.cursor = (self.disp.shape[1]//2, self.disp.shape[0]//2)

        cv2.setMouseCallback(self.window, self._on_mouse)

    def _to_orig(self, x_disp: int, y_disp: int) -> Tuple[int, int]:
        x = int(round(x_disp / self.scale))
        y = int(round(y_disp / self.scale))
        x = max(0, min(self.w0-1, x))
        y = max(0, min(self.h0-1, y))
        return x, y

    def _to_disp(self, x_orig: int, y_orig: int) -> Tuple[int, int]:
        x = int(round(x_orig * self.scale))
        y = int(round(y_orig * self.scale))
        return x, y

    def _draw(self):
        self.overlay = self.disp.copy()
        # Draw existing points with indices
        for i, (xo, yo) in enumerate(self.points):
            xd, yd = self._to_disp(xo, yo)
            cv2.circle(self.overlay, (xd, yd), 6, (0, 255, 0), -1)
            cv2.putText(self.overlay, str(i+1), (xd+8, yd-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Zoom inset
        if self.zoom_on and self.cursor is not None:
            cx_d, cy_d = self.cursor
            cx, cy = self._to_orig(cx_d, cy_d)
            r = self.zoom_box
            x0, y0 = max(0, cx-r), max(0, cy-r)
            x1, y1 = min(self.w0, cx+r), min(self.h0, cy+r)
            roi = self.img_orig[y0:y1, x0:x1]
            if roi.size > 0:
                zoom = cv2.resize(roi, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_NEAREST)
                zh, zw = zoom.shape[:2]
                # place inset at top-left with border
                inset_pad = 10
                ix0, iy0 = inset_pad, inset_pad
                ix1, iy1 = ix0+zw, iy0+zh
                # clip if larger than display
                if ix1 > self.overlay.shape[1] or iy1 > self.overlay.shape[0]:
                    # reduce zoom if necessary
                    scale_fit = min((self.overlay.shape[1]-2*inset_pad)/zw, (self.overlay.shape[0]-2*inset_pad)/zh, 1.0)
                    if scale_fit < 1.0:
                        zoom = cv2.resize(zoom, None, fx=scale_fit, fy=scale_fit, interpolation=cv2.INTER_NEAREST)
                        zh, zw = zoom.shape[:2]
                        ix1, iy1 = ix0+zw, iy0+zh
                # draw frame
                cv2.rectangle(self.overlay, (ix0-1, iy0-1), (ix1+1, iy1+1), (255,255,255), 2)
                self.overlay[iy0:iy1, ix0:ix1] = zoom
                # crosshair at center of inset corresponds to cursor
                cv2.drawMarker(self.overlay, (ix0+zw//2, iy0+zh//2), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        # Instructions
        cv2.putText(self.overlay, f"Points: {len(self.points)}/{self.needed}  [Z] toggle zoom  [+/-] zoom level  [ESC] cancel  press any key to accept when {self.needed}",
                    (10, self.overlay.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.needed:
            xo, yo = self._to_orig(x, y)
            self.points.append((xo, yo))

    def collect(self) -> List[Tuple[int, int]]:
        while True:
            self._draw()
            cv2.imshow(self.window, self.overlay)
            key = cv2.waitKey(20) & 0xFF
            if key != 255:  # a key pressed
                if key in (27,):  # ESC
                    cv2.destroyWindow(self.window)
                    return self.points
                elif key in (ord('z'), ord('Z')):
                    self.zoom_on = not self.zoom_on
                elif key in (ord('+'), ord('=')):
                    self.zoom_factor = min(10.0, self.zoom_factor * 1.25)
                elif key in (ord('-'), ord('_')):
                    self.zoom_factor = max(1.0, self.zoom_factor / 1.25)
                else:
                    if len(self.points) >= self.needed:
                        break
        cv2.destroyWindow(self.window)
        return self.points


# ------------------------
# CSV / inline source points (6 points)
# ------------------------

def _read_csv_coords(csv_path: str, needed: int = 6) -> np.ndarray:
    import csv
    pts: List[Tuple[float, float]] = []
    with open(csv_path, 'r', newline='') as f:
        data = f.read()
        try:
            dialect = csv.Sniffer().sniff(data, delimiters=",;\t ")
            f.seek(0)
            rdr = csv.reader(f, dialect)
        except Exception:
            f.seek(0)
            rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 2:
                continue
            try:
                x = float(str(row[0]).strip())
                y = float(str(row[1]).strip())
                pts.append((x, y))
            except ValueError:
                continue
    if len(pts) < needed:
        raise ValueError(f"CSV needs at least {needed} numeric rows of x,y.")
    if len(pts) > needed:
        pts = pts[:needed]
    return np.array(pts, dtype=np.float32)


def _src_points(coords_str: Optional[str], csv_path: Optional[str], needed: int = 6) -> np.ndarray:
    if coords_str:
        pts: List[Tuple[float, float]] = []
        parts = [p for p in coords_str.split(';') if p.strip()]
        if len(parts) != needed:
            raise ValueError(f"Need exactly {needed} source coordinates.")
        for token in parts:
            x_str, y_str = token.split(',')
            pts.append((float(x_str.strip()), float(y_str.strip())))
        return np.array(pts, dtype=np.float32)
    elif csv_path:
        return _read_csv_coords(csv_path, needed=needed)
    else:
        raise ValueError("No source coordinates provided.")


# ------------------------
# Stats / diagnostics
# ------------------------

def _reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_h = cv2.convertPointsToHomogeneous(src.reshape(-1,1,2)).reshape(-1,3)
    proj = (src_h @ H.T)
    proj = proj[:, :2] / proj[:, 2:3]
    err = dst - proj
    d = np.sqrt(np.sum(err**2, axis=1))
    return d, proj


def _print_stats(dists: np.ndarray) -> None:
    rmse = float(np.sqrt(np.mean(dists**2)))
    med = float(np.median(dists))
    p95 = float(np.percentile(dists, 95))
    mx = float(np.max(dists))
    print("Reprojection error stats (pixels):")
    print(" per-point:", np.round(dists, 3).tolist())
    print(f" RMSE: {rmse:.3f}  median: {med:.3f}  p95: {p95:.3f}  max: {mx:.3f}")


def _draw_diagnostics(base_bgr: np.ndarray, clicked: np.ndarray, proj: np.ndarray, poly_h: np.ndarray) -> np.ndarray:
    img = base_bgr.copy()
    # error vectors
    for i in range(len(clicked)):
        p = tuple(np.round(clicked[i]).astype(int))
        q = tuple(np.round(proj[i]).astype(int))
        cv2.circle(img, p, 5, (0, 0, 255), -1)    # clicked = red
        cv2.circle(img, q, 5, (0, 255, 0), -1)    # projected = green
        cv2.arrowedLine(img, q, p, (0, 255, 255), 2, tipLength=0.3)  # vector from proj→clicked
        cv2.putText(img, str(i+1), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # warped polygon overlay
    pts_i32 = np.round(poly_h).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_i32], (255, 255, 255))
    img = cv2.addWeighted(img, 0.75, overlay, 0.25, 0)
    return img


# ------------------------
# Core pipeline
# ------------------------

def run(image_path: str, dxf_path: str, out_mask_path: str, *, layer: Optional[str] = None,
        line_snap_tol: float = 1e-3, src_corners: Optional[str] = None,
        src_csv: Optional[str] = None, save_diag: Optional[str] = None,
        use_ransac: bool = False, ransac_thresh: float = 4.0, save_overlay: Optional[str] = None,
        save_config: Optional[str] = None) -> None:
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    # Read DXF polygon
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    poly = _largest_closed_poly_from_msp(msp, layer=layer, line_snap_tol=line_snap_tol)

    # Source points (6 in DXF units)
    src = _src_points(src_corners, src_csv, needed=6)

    # Destination points (6 clicks)
    print("Click 6 destination points on the image in the SAME order as the CSV.")
    clicks = ClickCollector(img, needed=6).collect()
    if len(clicks) != 6:
        raise RuntimeError("Need exactly 6 image points.")
    dst = np.array(clicks, dtype=np.float32)

    # Optional: save configuration (src + dst + settings) to a JSON/TXT file
    if save_config:
        try:
            cfg = {
                "version": 1,
                "src_points": src.tolist(),          # DXF coordinates (source)
                "image_points": dst.tolist(),        # the 6 clicked image points
                "layer": layer,
                "tolerance": float(line_snap_tol),
                "use_ransac": bool(use_ransac),
                "ransac_thresh": float(ransac_thresh),
            }
            with open(save_config, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"Saved configuration to: {save_config}")
        except Exception as e:
            print(f"Warning: could not save configuration file '{save_config}': {e}")

    # Fit homography
    method = cv2.RANSAC if use_ransac else 0
    H, mask_inliers = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("Homography computation failed.")

    # Residuals & stats
    dists, proj = _reprojection_errors(H, src, dst)
    _print_stats(dists)

    # Transform polygon and rasterize mask
    poly_h = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H).reshape(-1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly_h).astype(np.int32)], 255)
    ok = cv2.imwrite(out_mask_path, mask)
    if not ok:
        raise IOError(f"Could not write mask to {out_mask_path}")
    print(f"Saved mask: {out_mask_path}")

    # Overlay (mask on original image)
    if save_overlay:
        overlay = img.copy()
        alpha = 0.35  # transparency of the white fill
        mask_bool = mask > 0
        if np.any(mask_bool):
            overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * 255).astype(np.uint8)
        ok3 = cv2.imwrite(save_overlay, overlay)
        if ok3:
            print(f"Saved overlay: {save_overlay}")
        else:
            print("Warning: failed to save overlay image")

    # Diagnostics image
    if save_diag:
        diag = _draw_diagnostics(img, dst, proj, poly_h)
        ok2 = cv2.imwrite(save_diag, diag)
        if ok2:
            print(f"Saved diagnostics: {save_diag}")
        else:
            print("Warning: failed to save diagnostics image")


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DXF→Image mask with 6-point statistical homography, overlay, and reprojection diagnostics.")
    parser.add_argument("--image", default=None, help="Path to street image (JPG/PNG). If omitted, a file dialog will open.")
    parser.add_argument("--dxf", default=None, help="Path to DXF file. If omitted, a file dialog will open.")
    parser.add_argument("--out", default=None, help="Path to save the black/white mask PNG. If omitted, a save dialog will open.")
    parser.add_argument("--layer", default=None, help="Only consider entities on this DXF layer name.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Endpoint snap tolerance for LINE chains (drawing units).")
    parser.add_argument("--src-csv", default=None, help="CSV/TXT with 6 rows of x,y in DXF units (header allowed).")
    parser.add_argument("--src-corners", default=None, help="Inline: 'x1,y1;...;x6,y6' (same order as image clicks).")
    parser.add_argument("--save-diagnostics", default=None, help="PNG path for the error visualization overlay (arrows + warped polygon).")
    parser.add_argument("--save-overlay", default=None, help="PNG path for the mask-over-photo visual control.")
    parser.add_argument("--ransac", action="store_true", help="Use RANSAC when fitting H (helps with a single bad click).")
    parser.add_argument("--ransac-thresh", type=float, default=4.0, help="RANSAC reprojection threshold in pixels.")
    parser.add_argument("--save-config", default=None,
                        help="Optional path to save a config TXT/JSON with src & image points.")
    args = parser.parse_args()

    # File pickers
    if (args.image is None or args.dxf is None or args.out is None or args.save_overlay is None):
        if not _TK_OK:
            raise RuntimeError("tkinter not available. Please pass --image/--dxf/--out explicitly, or install tkinter.")
        root = tk.Tk(); root.withdraw()
        if args.image is None:
            args.image = filedialog.askopenfilename(title="Select street image", filetypes=[("Images", ".png .jpg .jpeg .bmp .tif .tiff"), ("All files", "*.*")])
        if args.dxf is None:
            args.dxf = filedialog.askopenfilename(title="Select DXF file", filetypes=[("DXF", ".dxf"), ("All files", "*.*")])
        if args.out is None:
            args.out = filedialog.asksaveasfilename(title="Save output MASK as", defaultextension=".png", filetypes=[("PNG", ".png"), ("All files", "*.*")])
        # NEW: ask for overlay path (optional). User can Cancel to skip.
        if args.save_overlay is None:
            suggested = None
            try:
                import os
                if args.out:
                    base, _ = os.path.splitext(args.out)
                    suggested = base + "_overlay.png"
            except Exception:
                suggested = None
            args.save_overlay = filedialog.asksaveasfilename(
                title="Save OVERLAY (mask on photo) as — Cancel to skip",
                initialfile=(suggested if suggested else "overlay.png"),
                defaultextension=".png",
                filetypes=[("PNG", ".png"), ("All files", "*.*")]
            )
            if not args.save_overlay:
                args.save_overlay = None

        # Ask where to save CONFIG (optional; user can cancel to skip)
        if args.save_config is None:
            import os
            suggested_cfg = None
            try:
                if args.out:
                    base, _ = os.path.splitext(args.out)
                    suggested_cfg = base + "_config.json"
            except Exception:
                suggested_cfg = None
            args.save_config = filedialog.asksaveasfilename(
                title="Save CONFIG (src & image points) as — Cancel to skip",
                initialfile=(suggested_cfg if suggested_cfg else "config.json"),
                defaultextension=".json",
                filetypes=[("JSON files", ".json"), ("All files", "*.*")]
            )
            if not args.save_config:
                args.save_config = None
        root.destroy()

    # If no coords provided, open dialog for CSV now
    if (not args.src_corners and not args.src_csv):
        if not _TK_OK:
            raise SystemExit("You must provide 6 DXF coordinates via --src-corners or --src-csv.")
        root = tk.Tk(); root.withdraw()
        args.src_csv = filedialog.askopenfilename(title="Select CSV/TXT with 6 DXF coordinates (x,y)", filetypes=[("CSV/TXT", ".csv .txt"), ("All files", "*.*")])
        root.destroy()

    if not args.image or not args.dxf or not args.out:
        raise SystemExit("Image, DXF, and output paths are required. Either provide CLI args or choose files in the dialogs.")

    run(
        image_path=args.image,
        dxf_path=args.dxf,
        out_mask_path=args.out,
        layer=args.layer,
        line_snap_tol=args.tolerance,
        src_corners=args.src_corners,
        src_csv=args.src_csv,
        save_diag=args.save_diagnostics,
        use_ransac=args.ransac,
        ransac_thresh=args.ransac_thresh,
        save_overlay=args.save_overlay,
        save_config=args.save_config,
    )
