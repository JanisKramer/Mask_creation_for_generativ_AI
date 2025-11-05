#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree Masks — 2 real calibration rigs (ground L + vertical 1m), line-distance interpolation
with SAFE extrapolation (no tiny last tree).

Flow:
1. 4 clicks: plane (BL, TL, TR, BR)
2. 4 clicks: Rig 1  -> G, F, S, V
3. 4 clicks: Rig 2  -> G, F, S, V
4. 2 clicks: tree line (start → end)
5. Export: mask only
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import gradio as gr
import numpy as np

EPS = 1e-6

# ----------------- small helpers -----------------
def to_rgb(img):
    if img is None: return None
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def order_quad_bl_tl_tr_br(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] != 4:
        return pts
    idx = np.lexsort((pts[:,0], pts[:,1]))
    top2 = pts[idx[:2]]
    bot2 = pts[idx[2:]]
    tl, tr = sorted(top2, key=lambda p: p[0])
    bl, br = sorted(bot2, key=lambda p: p[0])
    return np.array([bl, tl, tr, br], dtype=np.float32)

def safe_homography(src: np.ndarray, dst: np.ndarray):
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None or not np.isfinite(H).all(): return None
    if abs(H[2,2]) > EPS: H = H / H[2,2]
    return H

def extend_quad_to_image_bottom(quad_bl_tl_tr_br: np.ndarray, img_h: int) -> np.ndarray:
    def line_intersect_y(p0, p1, y):
        x0,y0 = p0; x1,y1 = p1
        t = 0 if abs(y1-y0)<EPS else (y-y0)/(y1-y0)
        return np.array([x0 + t*(x1-x0), y], dtype=np.float32)
    bl, tl, tr, br = quad_bl_tl_tr_br
    yb = float(img_h - 1)
    bl_ext = line_intersect_y(tl, bl, yb)
    br_ext = line_intersect_y(tr, br, yb)
    return np.array([bl_ext, tl, tr, br_ext], dtype=np.float32)

def to_topdown(pt_xy: Tuple[float,float], H: np.ndarray) -> np.ndarray:
    p = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    q = H @ p; q /= (q[2] + EPS)
    return q[:2].astype(np.float32)

def from_topdown(pt_xy: Tuple[float,float], Hinv: np.ndarray) -> np.ndarray:
    p = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    q = Hinv @ p; q /= (q[2] + EPS)
    return q[:2].astype(np.float32)

def seg_len(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def project_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-9:
        return 0.0
    ap = p - a
    t = float(np.dot(ap, ab) / ab2)
    return max(0.0, min(1.0, t))

# ----------------- sprite loading -----------------
def load_tree_sprite(path: str, force_luma_alpha: bool = True, thresh: int = 10):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"tree sprite not found: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:,:,3] = alpha
        bgra[:,:,3] = (bgra[:,:,3] > 127).astype(np.uint8) * 255
        return bgra

    b,g,r,a = cv2.split(img)
    opaque_ratio = (a >= 250).sum() / a.size
    alpha_dynamic = int(a.max()) - int(a.min())
    if force_luma_alpha or (opaque_ratio > 0.98 or alpha_dynamic < 5):
        gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY)
        a = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        img[:,:,3] = a

    img[:,:,3] = (img[:,:,3] > 127).astype(np.uint8) * 255
    return img

# ----------------- params -----------------
@dataclass
class Params:
    spacing_m: float = 10.0
    count_override: Optional[int] = None
    tree_height_m: float = 3.0
    global_scale: float = 1.0
    max_px: int = 800
    min_px: int = 2

# ----------------- app -----------------
class TwoRigTrees:
    def __init__(self, sprite_path="tree.png"):
        self.sprite_bgra = load_tree_sprite(sprite_path, force_luma_alpha=True)
        self.sprite_path = sprite_path

        self.img_bgr: Optional[np.ndarray] = None
        self.mode = "plane"

        self.clicks_plane: List[Tuple[float,float]] = []
        self.clicks_rig1: List[Tuple[float,float]] = []   # G, F, S, V
        self.clicks_rig2: List[Tuple[float,float]] = []   # G, F, S, V
        self.clicks_line: List[Tuple[float,float]] = []

        # homography
        self.pad_x, self.pad_y = 200, 200
        self.top_w, self.top_h = 3000, 2000
        self.H = None
        self.Hinv = None
        self.ext_quad = None

        # topdown bases of rigs
        self.rig1_top: Optional[np.ndarray] = None
        self.rig2_top: Optional[np.ndarray] = None

        # vertical ppm measured in IMAGE at rigs
        self.rig1_vert_ppm: Optional[float] = None
        self.rig2_vert_ppm: Optional[float] = None

        # line in topdown
        self.line_top: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.line_length_m: Optional[float] = None
        self.m_per_px_top: Optional[float] = None

        # line-distance for each rig (0..1 along line)
        self.rig1_t_on_line: Optional[float] = None
        self.rig2_t_on_line: Optional[float] = None

        self.params = Params()

    # ---------- helpers ----------
    def _calc_global_scale_from_rig1(self):
        if self.H is None or len(self.clicks_rig1) < 2:
            return
        g1 = self.clicks_rig1[0]
        f1 = self.clicks_rig1[1]
        g1t = to_topdown(g1, self.H)
        f1t = to_topdown(f1, self.H)
        px = seg_len((f1t - g1t).astype(np.float64))
        if px > 1e-3:
            self.m_per_px_top = 1.0 / px
        else:
            self.m_per_px_top = None

    def _interp_vert_ppm_along_line(self, t_line: float) -> float:
        if self.rig1_vert_ppm is None and self.rig2_vert_ppm is None:
            return 100.0
        if self.rig1_vert_ppm is not None and self.rig2_vert_ppm is not None:
            t1, t2 = self.rig1_t_on_line, self.rig2_t_on_line
            v1, v2 = self.rig1_vert_ppm, self.rig2_vert_ppm
            if t_line <= t1:   # before near → just near
                return v1
            if t_line >= t2:   # after far → just far
                return v2
            # between
            alpha = (t_line - t1) / (t2 - t1)
            return v1 + alpha * (v2 - v1)
        # only one rig:
        return self.rig1_vert_ppm or self.rig2_vert_ppm



    # ---------- plane ----------
    def lock_plane(self):
        quad = order_quad_bl_tl_tr_br(np.array(self.clicks_plane, dtype=np.float32))
        self.ext_quad = extend_quad_to_image_bottom(quad, self.img_bgr.shape[0])
        dst = np.array([
            [self.pad_x,              self.top_h-1-self.pad_y],
            [self.pad_x,              self.pad_y],
            [self.top_w-1-self.pad_x, self.pad_y],
            [self.top_w-1-self.pad_x, self.top_h-1-self.pad_y],
        ], dtype=np.float32)
        H = safe_homography(self.ext_quad, dst)
        if H is None:
            raise RuntimeError("Homography failed.")
        Hinv = np.linalg.inv(H); Hinv /= (Hinv[2,2] + EPS)
        self.H, self.Hinv = H, Hinv
        self.mode = "rig1"

    # ---------- rigs ----------
    def _process_rig_clicks(self, rig_clicks: List[Tuple[float,float]]):
        G, F, S, V = rig_clicks
        base_top = to_topdown(G, self.H)
        v_ppm = seg_len((np.array(V) - np.array(G)).astype(np.float64)) / 1.0
        return base_top, v_ppm

    # ---------- line ----------
    def _lock_line(self):
        l0t = to_topdown(self.clicks_line[0], self.H)
        l1t = to_topdown(self.clicks_line[1], self.H)
        self.line_top = (l0t, l1t)

        if self.m_per_px_top is not None:
            L_px_top = seg_len((l1t - l0t).astype(np.float64))
            self.line_length_m = L_px_top * self.m_per_px_top
        else:
            self.line_length_m = None

        # project rigs to line (0..1)
        if self.rig1_top is not None:
            self.rig1_t_on_line = project_point_on_segment(self.rig1_top, l0t, l1t)
        if self.rig2_top is not None:
            self.rig2_t_on_line = project_point_on_segment(self.rig2_top, l0t, l1t)

        self.mode = "ready"

    # ---------- stamping ----------
    def stamp_to_mask(self, mask: np.ndarray, center_bottom_xy: Tuple[int,int], sprite_bgra: np.ndarray):
        x, y = int(round(center_bottom_xy[0])), int(round(center_bottom_xy[1]))
        sh, sw = sprite_bgra.shape[:2]
        tl_x = x - sw//2
        tl_y = y - sh

        Hh, Ww = mask.shape[:2]
        x0 = max(0, tl_x); y0 = max(0, tl_y)
        x1 = min(Ww, tl_x + sw); y1 = min(Hh, tl_y + sh)
        if x0 >= x1 or y0 >= y1: return

        sx0 = x0 - tl_x; sy0 = y0 - tl_y
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)

        alpha = sprite_bgra[sy0:sy1, sx0:sx1, 3]
        mask[y0:y1, x0:x1] = np.where(alpha >= 128, 255, mask[y0:y1, x0:x1])

    # ---------- preview ----------
    def preview(self) -> np.ndarray:
        vis = self.img_bgr.copy()

        # plane
        if len(self.clicks_plane) > 0:
            for (px, py) in self.clicks_plane:
                cv2.circle(vis, (int(px), int(py)), 5, (0,165,255), -1)
        if len(self.clicks_plane) >= 4:
            quad = order_quad_bl_tl_tr_br(np.array(self.clicks_plane[:4], dtype=np.float32))
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0,165,255), 2)
            if self.ext_quad is not None:
                cv2.polylines(vis, [self.ext_quad.astype(np.int32)], True, (0,255,0), 1)

        # rig1
        if len(self.clicks_rig1) > 0:
            for p in self.clicks_rig1:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (255,0,255), -1)
            if len(self.clicks_rig1) == 4:
                G,F,S,V = self.clicks_rig1
                cv2.line(vis, (int(G[0]),int(G[1])), (int(F[0]),int(F[1])), (255,0,255), 2)
                cv2.line(vis, (int(G[0]),int(G[1])), (int(S[0]),int(S[1])), (255,0,255), 2)
                cv2.line(vis, (int(G[0]),int(G[1])), (int(V[0]),int(V[1])), (0,0,255), 2)
                if self.rig1_vert_ppm is not None:
                    cv2.putText(vis, f"R1 {self.rig1_vert_ppm:.1f}px/m", (int(G[0])+10, int(G[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

        # rig2
        if len(self.clicks_rig2) > 0:
            for p in self.clicks_rig2:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0,0,255), -1)
            if len(self.clicks_rig2) == 4:
                G,F,S,V = self.clicks_rig2
                cv2.line(vis, (int(G[0]),int(G[1])), (int(F[0]),int(F[1])), (0,0,255), 2)
                cv2.line(vis, (int(G[0]),int(G[1])), (int(S[0]),int(S[1])), (0,0,255), 2)
                cv2.line(vis, (int(G[0]),int(G[1])), (int(V[0]),int(V[1])), (0,255,255), 2)
                if self.rig2_vert_ppm is not None:
                    cv2.putText(vis, f"R2 {self.rig2_vert_ppm:.1f}px/m", (int(G[0])+10, int(G[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        # line
        if len(self.clicks_line) == 1:
            cv2.circle(vis, (int(self.clicks_line[0][0]), int(self.clicks_line[0][1])), 5, (255,0,0), -1)
        elif len(self.clicks_line) == 2:
            p0, p1 = self.clicks_line
            cv2.line(vis, (int(p0[0]),int(p0[1])), (int(p1[0]),int(p1[1])), (255,0,0), 2)
            if self.m_per_px_top is not None and self.H is not None:
                l0t = to_topdown(p0, self.H); l1t = to_topdown(p1, self.H)
                L_px_top = seg_len((l1t - l0t).astype(np.float64))
                L_m = L_px_top * self.m_per_px_top
                mid = (int((p0[0]+p1[0])/2), int((p0[1]+p1[1])/2))
                spacing = max(float(self.params.spacing_m), 1e-6)
                est_n = int(np.floor(L_m / spacing)) + 1
                cv2.putText(vis, f"{L_m:.2f} m  • ~{est_n} trees",
                            (mid[0]+10, mid[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

        return vis

    # ---------- build ----------
    def build_masks(self):
        assert self.mode == "ready"
        h, w = self.img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        l0, l1 = self.line_top
        v = (l1 - l0).astype(np.float64)
        L = seg_len(v); vn = v / (L + EPS)

        if self.m_per_px_top is not None:
            L_m = L * self.m_per_px_top
        else:
            L_m = L

        spacing = max(float(self.params.spacing_m), 1e-6)
        count_override = self.params.count_override

        if count_override and count_override > 0:
            n = int(count_override)
            s_m = np.linspace(0.0, L_m, n, endpoint=True)
        else:
            n_steps = int(np.floor(L_m / spacing))
            s_m = np.arange(n_steps + 1, dtype=np.float64) * spacing

        # t-values along the line 0..1
        t_vals = (s_m / (L_m + EPS)).astype(np.float64)

        # dedupe image positions
        pts = []
        seen = set()
        for t in t_vals:
            base_top = l0 + vn * (t * L)
            base_img = from_topdown(base_top, self.Hinv)
            key = (int(round(base_img[0])), int(round(base_img[1])))
            if key in seen:
                continue
            seen.add(key)
            pts.append((t, base_top, base_img))

        for t_line, base_top, base_img in pts:
            vert_ppm = self._interp_vert_ppm_along_line(t_line)
            tree_px_h = vert_ppm * self.params.tree_height_m * self.params.global_scale
            tree_px_h = int(np.clip(round(tree_px_h), self.params.min_px, self.params.max_px))

            sh, sw = self.sprite_bgra.shape[:2]
            scale = tree_px_h / float(sh)
            new_w = max(2, int(round(sw * scale)))
            new_h = max(2, int(round(sh * scale)))

            rgb  = self.sprite_bgra[..., :3]
            alpha = self.sprite_bgra[..., 3]
            rgb_scaled  = cv2.resize(rgb,  (new_w, new_h), interpolation=cv2.INTER_AREA)
            alpha_scaled = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            alpha_scaled = (alpha_scaled >= 128).astype(np.uint8) * 255
            sprite_scaled = np.dstack([rgb_scaled, alpha_scaled])

            self.stamp_to_mask(mask, (base_img[0], base_img[1]), sprite_scaled)

        return mask, len(pts)

# ----------------- Gradio UI -----------------
SPRITE_PATH = "tree.png"
app = TwoRigTrees(sprite_path=SPRITE_PATH)

def ui_load(img, img_info):
    app.__init__(sprite_path=SPRITE_PATH)
    app.img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return to_rgb(app.preview()), "Click 4 points (BL, TL, TR, BR)."

def ui_click(evt: gr.SelectData):
    x, y = float(evt.index[0]), float(evt.index[1])

    if app.mode == "plane":
        app.clicks_plane.append((x,y))
        if len(app.clicks_plane) == 4:
            try:
                app.lock_plane()
                return to_rgb(app.preview()), "Plane locked. Click RIG 1: G, F, S, V."
            except Exception as e:
                app.clicks_plane = []
                return to_rgb(app.preview()), str(e)
        return to_rgb(app.preview()), f"Plane {len(app.clicks_plane)}/4…"

    elif app.mode == "rig1":
        app.clicks_rig1.append((x,y))
        if len(app.clicks_rig1) == 4:
            base_top, vppm = app._process_rig_clicks(app.clicks_rig1)
            app.rig1_top = base_top
            app.rig1_vert_ppm = vppm
            app._calc_global_scale_from_rig1()
            app.mode = "rig2"
            return to_rgb(app.preview()), "RIG 1 set. Click RIG 2: G, F, S, V."
        return to_rgb(app.preview()), f"RIG 1 {len(app.clicks_rig1)}/4…"

    elif app.mode == "rig2":
        app.clicks_rig2.append((x,y))
        if len(app.clicks_rig2) == 4:
            base_top, vppm = app._process_rig_clicks(app.clicks_rig2)
            app.rig2_top = base_top
            app.rig2_vert_ppm = vppm
            app.mode = "line"
            return to_rgb(app.preview()), "RIG 2 set. Click 2 points for the tree line."
        return to_rgb(app.preview()), f"RIG 2 {len(app.clicks_rig2)}/4…"

    elif app.mode == "line":
        app.clicks_line.append((x,y))
        if len(app.clicks_line) == 2:
            app._lock_line()
            return to_rgb(app.preview()), "Line set. Adjust params, then Export."
        return to_rgb(app.preview()), "Line 1/2…"

    return to_rgb(app.preview()), ""

def ui_params(spacing_m, count_override, tree_h_m, global_scale, max_px, min_px):
    app.params.spacing_m = float(spacing_m)
    app.params.count_override = int(count_override) if (count_override and count_override>0) else None
    app.params.tree_height_m = float(tree_h_m)
    app.params.global_scale = float(global_scale)
    app.params.max_px = int(max_px)
    app.params.min_px = int(min_px)
    return to_rgb(app.preview()), "Params updated."

def ui_export(out_dir):
    if app.mode != "ready":
        return None, "Finish plane → rig1 → rig2 → line first."
    os.makedirs(out_dir, exist_ok=True)
    mask, n = app.build_masks()
    path = os.path.join(out_dir, "trees_mask.png")
    cv2.imwrite(path, mask)
    return to_rgb(app.preview()), f"Saved {path} with {n} trees."

with gr.Blocks(title="Tree Masks — 2 rigs, SAFE line-distance interpolation") as demo:
    gr.Markdown("Flow: 4x plane → 4x rig1 → 4x rig2 → 2x line → export. \
                 \nSizes are interpolated **along the line** and **safely extrapolated** so far trees don't collapse to 1 px.")

    img_in = gr.Image(label="Input (click)", interactive=True)
    status = gr.Markdown("")
    img_in.upload(fn=ui_load, inputs=[img_in, img_in], outputs=[img_in, status])
    img_in.select(fn=ui_click, outputs=[img_in, status])

    with gr.Row():
        spacing_m   = gr.Number(label="Spacing (m)", value=10.0)
        count_opt   = gr.Number(label="Count (optional)", value=None)
    with gr.Row():
        tree_h_m    = gr.Number(label="Tree height (m)", value=3.0)
        gscale      = gr.Slider(label="Global scale", minimum=0.3, maximum=3.0, step=0.05, value=1.0)
        max_px      = gr.Number(label="Max px", value=800)
        min_px      = gr.Number(label="Min px", value=2)

    apply_btn = gr.Button("Apply params")
    apply_btn.click(fn=ui_params,
                    inputs=[spacing_m, count_opt, tree_h_m, gscale, max_px, min_px],
                    outputs=[img_in, status])

    out_dir = gr.Textbox(label="Export folder", value="export_trees_masks")
    export_btn = gr.Button("Export Masks")
    export_btn.click(fn=ui_export, inputs=[out_dir], outputs=[img_in, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_api=False, show_error=True)
