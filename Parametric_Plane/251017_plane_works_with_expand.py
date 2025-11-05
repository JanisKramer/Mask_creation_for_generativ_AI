#!/usr/bin/env python3
"""
Street Bands — robust bottom-extended version (v4.1)

Changes from v4 (per your request):
- **No slow image refresh for the first 3 plane clicks.** The image only updates after the 4th click (when the plane is ready). After the plane is locked, live previews remain as before.
- **Baseline and reference are chosen via image clicks, not coordinates.**
  Flow: 4 clicks for plane → 2 clicks for baseline → 2 clicks for reference (+ length in cm).
- Still extends **only to the bottom** of the image (TL/TR fixed).
- Exports per‑band **black/white** masks.
"""
from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import cv2
import gradio as gr

# ---------------------------
# Utility & math helpers
# ---------------------------
EPS = 1e-6


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def safe_homography(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    try:
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if H is None or not np.isfinite(H).all():
            return None
        if abs(H[2, 2]) > EPS:
            H = H / H[2, 2]
        return H
    except Exception:
        return None


def order_quad_bl_tl_tr_br(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    idx = np.lexsort((pts[:, 0], pts[:, 1]))
    top2 = pts[idx[:2]]
    bot2 = pts[idx[2:]]
    tl, tr = sorted(top2, key=lambda p: p[0])
    bl, br = sorted(bot2, key=lambda p: p[0])
    return np.array([bl, tl, tr, br], dtype=np.float32)


def line_intersect_y(p0: np.ndarray, p1: np.ndarray, y_val: float) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    if abs(y1 - y0) < EPS:
        return np.array([x1, y_val], dtype=np.float32)
    t = (y_val - y0) / (y1 - y0)
    x = x0 + t * (x1 - x0)
    return np.array([x, y_val], dtype=np.float32)


def extend_quad_to_image_bottom(quad_bl_tl_tr_br: np.ndarray, img_h: int) -> np.ndarray:
    bl, tl, tr, br = quad_bl_tl_tr_br
    y_bottom = float(img_h - 1)
    bl_ext = line_intersect_y(tl, bl, y_bottom)
    br_ext = line_intersect_y(tr, br, y_bottom)
    return np.array([bl_ext, tl, tr, br_ext], dtype=np.float32)


def build_homography_padded(quad_img: np.ndarray, pad_x: int, pad_y: int, top_w: int, top_h: int):
    dst = np.array([
        [pad_x, top_h - 1 - pad_y],
        [pad_x, pad_y],
        [top_w - 1 - pad_x, pad_y],
        [top_w - 1 - pad_x, top_h - 1 - pad_y],
    ], dtype=np.float32)
    H = safe_homography(quad_img.astype(np.float32), dst)
    if H is None:
        raise ValueError("Homography failed; check clicked plane and image quality.")
    Hinv = np.linalg.inv(H)
    Hinv /= (Hinv[2, 2] + EPS)
    return H, Hinv


def to_topdown(pt_xy: Tuple[float, float], H: np.ndarray) -> np.ndarray:
    p = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    q = H @ p
    q /= (q[2] + EPS)
    return q[:2].astype(np.float32)


def seg_len(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


@dataclass
class Band:
    label: str
    from_cm: float
    to_cm: float
    color_rgb: Tuple[int, int, int]


PALETTE = [
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (168, 218, 220), (241, 250, 238),
    (244, 162, 97), (233, 196, 106), (42, 157, 143), (38, 70, 83), (187, 62, 3)
]


class StreetBands:
    def __init__(self):
        self.img_bgr: Optional[np.ndarray] = None
        self.img_path: Optional[str] = None

        # Click state machine
        self.mode: str = 'plane'  # 'plane' -> 'baseline' -> 'reference' -> 'bands'
        self.clicks_plane: List[Tuple[float, float]] = []  # 4
        self.clicks_baseline: List[Tuple[float, float]] = []  # 2
        self.clicks_ref: List[Tuple[float, float]] = []  # 2

        # Geom & bands
        self.bands: List[Band] = []
        self.invert_normal: bool = False

        self.pad_x, self.pad_y = 200, 200
        self.top_w, self.top_h = 3000, 2000
        self.H: Optional[np.ndarray] = None
        self.Hinv: Optional[np.ndarray] = None
        self.quad_img_raw: Optional[np.ndarray] = None
        self.quad_img_extended: Optional[np.ndarray] = None
        self.m_per_px: Optional[float] = None
        self.baseline_top: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self.label_top: Optional[np.ndarray] = None  # int32 label map

        # Reference length (cm)
        self.ref_cm: float = 100.0

    # ---------------
    # Image loading / reset
    # ---------------
    def load_image(self, img: np.ndarray, img_path: str | None = None):
        self.__init__()
        self.img_bgr = img.copy()
        self.img_path = img_path
        return to_rgb_uint8(self.img_bgr), "Click 4 points (BL, TL, TR, BR). No preview will update until the 4th click."

    def start_over(self):
        if self.img_bgr is None:
            return None, "Load an image first."
        img = self.img_bgr.copy()
        self.__init__()
        self.img_bgr = img
        return to_rgb_uint8(self.img_bgr), "Reset. Click 4 points (BL, TL, TR, BR)."

    # ---------------
    # Plane clicks (no visual refresh until 4th)
    # ---------------
    def click_image(self, x: float, y: float):
        if self.img_bgr is None:
            return gr.update(), "Load an image first."

        if self.mode == 'plane':
            if len(self.clicks_plane) < 4:
                self.clicks_plane.append((x, y))
                # For the first 3 plane clicks, DO NOT update the image to avoid 2–3s delays
                if len(self.clicks_plane) < 4:
                    remaining = 4 - len(self.clicks_plane)
                    return gr.update(), f"Plane clicks: {len(self.clicks_plane)}/4. {remaining} to go…"
                # On the 4th click, lock plane and show preview
                return self.lock_plane()

        elif self.mode == 'baseline':
            self.clicks_baseline.append((x, y))
            if len(self.clicks_baseline) < 2:
                return self.preview(status="Baseline: 1/2 points. Pick the second point.")
            else:
                # Move to reference mode
                self.mode = 'reference'
                return self.preview(status="Baseline set. Now click 2 reference points (for known length).")

        elif self.mode == 'reference':
            self.clicks_ref.append((x, y))
            if len(self.clicks_ref) < 2:
                return self.preview(status="Reference: 1/2 points. Pick the second point.")
            else:
                # Compute baseline + scale and move to bands mode
                (b0x, b0y), (b1x, b1y) = self.clicks_baseline
                (r0x, r0y), (r1x, r1y) = self.clicks_ref
                return self.lock_baseline_and_reference(b0x, b0y, b1x, b1y, r0x, r0y, r1x, r1y, self.ref_cm)

        # bands mode: clicking the image has no special action (reserved for future tools)
        return self.preview()

    # ---------------
    # Plane + baseline + reference
    # ---------------
    def lock_plane(self):
        if self.img_bgr is None or len(self.clicks_plane) != 4:
            return gr.update(), "Click 4 points: BL, TL, TR, BR."
        h, w = self.img_bgr.shape[:2]
        quad = order_quad_bl_tl_tr_br(np.array(self.clicks_plane, dtype=np.float32))
        self.quad_img_raw = quad.copy()
        self.quad_img_extended = extend_quad_to_image_bottom(quad, h)
        try:
            self.H, self.Hinv = build_homography_padded(
                self.quad_img_extended, self.pad_x, self.pad_y, self.top_w, self.top_h
            )
        except ValueError as e:
            return to_rgb_uint8(self.img_bgr), str(e)
        self.mode = 'baseline'
        return self.preview(status="Plane locked. Now click 2 baseline points.")

    def lock_baseline_and_reference(self, bx0: float, by0: float, bx1: float, by1: float,
                                    rx0: float, ry0: float, rx1: float, ry1: float,
                                    ref_cm: float):
        if self.H is None:
            return gr.update(), "Lock the plane first."
        self.ref_cm = float(ref_cm)
        # Rectify
        b0t = to_topdown((bx0, by0), self.H)
        b1t = to_topdown((bx1, by1), self.H)
        r0t = to_topdown((rx0, ry0), self.H)
        r1t = to_topdown((ry1 if False else rx1, ry1), self.H)  # keep signature; no-op trick
        r1t = to_topdown((rx1, ry1), self.H)

        v = (b1t - b0t).astype(np.float64)
        if seg_len(v) < 2.0:
            return self.preview(status="Baseline too short. Pick two distinct points.")
        self.baseline_top = (b0t, b1t)

        ref_px = seg_len((r1t - r0t).astype(np.float64))
        if ref_px < 1.0:
            return self.preview(status="Reference segment too small after rectification.")
        ref_m = max(self.ref_cm, EPS) / 100.0
        self.m_per_px = ref_m / ref_px

        self.label_top = np.zeros((self.top_h, self.top_w), dtype=np.int32)
        self.mode = 'bands'
        return self.refresh_overlay(status=f"Baseline + scale locked (1 px ≈ {self.m_per_px:.4f} m). Add bands.")

    # ---------------
    # Bands
    # ---------------
    def add_band(self, label: str, from_cm: float, to_cm: float):
        if self.baseline_top is None or self.m_per_px is None:
            return self.preview(status="Lock baseline & reference first.")
        a, b = float(from_cm), float(to_cm)
        if a == b:
            return self.preview(status="Band has zero width. Use different from/to.")
        if a > b:
            a, b = b, a
        b0, b1 = self.baseline_top
        v = (b1 - b0).astype(np.float64)
        vn = v / (seg_len(v) + EPS)
        n = np.array([-vn[1], vn[0]], dtype=np.float64)
        if self.invert_normal:
            n = -n
        a_px = (a / 100.0) / self.m_per_px
        b_px = (b / 100.0) / self.m_per_px
        if abs(b_px - a_px) < 1.0:
            mid = 0.5 * (a_px + b_px)
            a_px, b_px = mid - 0.5, mid + 0.5
        L = seg_len(v) + 200.0
        u = vn * L * 0.5
        c = (b0 + b1) * 0.5
        p0 = c - u
        p1 = c + u
        q0 = p0 + n * a_px
        q1 = p1 + n * a_px
        r0 = p0 + n * b_px
        r1 = p1 + n * b_px
        poly = np.array([q0, q1, r1, r0], dtype=np.int32)
        if self.label_top is None:
            self.label_top = np.zeros((self.top_h, self.top_w), dtype=np.int32)
        band_idx = len(self.bands) + 1
        mask = np.zeros_like(self.label_top, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        self.label_top[mask > 0] = band_idx
        color = PALETTE[(band_idx - 1) % len(PALETTE)]
        self.bands.append(Band(label=label, from_cm=a, to_cm=b, color_rgb=color))
        return self.refresh_overlay(status=f"Added band '{label}' [{a:.1f}–{b:.1f} cm]")

    def delete_last_band(self):
        if not self.bands:
            return self.preview(status="No bands to delete.")
        self.bands.pop()
        self.rebuild_labels()
        return self.refresh_overlay(status="Deleted last band.")

    def clear_bands(self):
        self.bands.clear()
        if self.label_top is not None:
            self.label_top[:] = 0
        return self.refresh_overlay(status="Cleared all bands.")

    def rebuild_labels(self):
        if self.baseline_top is None or self.m_per_px is None:
            return
        self.label_top = np.zeros((self.top_h, self.top_w), dtype=np.int32)
        saved = self.bands.copy()
        self.bands = []
        for b in saved:
            self.add_band(b.label, b.from_cm, b.to_cm)

    # ---------------
    # Rendering & overlay
    # ---------------
    def _compose_preview_top(self) -> np.ndarray:
        if self.label_top is None:
            return np.zeros((self.top_h, self.top_w, 3), dtype=np.uint8)
        canvas = np.zeros((self.top_h, self.top_w, 3), dtype=np.uint8)
        for i, band in enumerate(self.bands, start=1):
            canvas[self.label_top == i] = band.color_rgb[::-1]
        return canvas

    def _warp_labels_to_image(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.img_bgr is None or self.Hinv is None:
            return None, None
        h, w = self.img_bgr.shape[:2]
        label_top_u16 = np.clip(self.label_top if self.label_top is not None else np.zeros((self.top_h, self.top_w), np.int32), 0, 65535).astype(np.uint16)
        label_img_u16 = cv2.warpPerspective(label_top_u16, self.Hinv, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        label_img = label_img_u16.astype(np.int32)
        overlay = np.zeros_like(self.img_bgr)
        for i, band in enumerate(self.bands, start=1):
            mask = (label_img == i)
            if np.any(mask):
                overlay[mask] = band.color_rgb[::-1]
        return label_img, overlay

    def preview(self, status: str = ""):
        if self.img_bgr is None:
            return gr.update(), "Load an image to start."
        vis = self.img_bgr.copy()
        # Draw only when plane is complete (to avoid slow refresh during first clicks)
        if len(self.clicks_plane) >= 4:
            quad = order_quad_bl_tl_tr_br(np.array(self.clicks_plane, dtype=np.float32))
            h = self.img_bgr.shape[0]
            ext = extend_quad_to_image_bottom(quad, h)
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 165, 255), 2)
            cv2.polylines(vis, [ext.astype(np.int32)], True, (0, 255, 0), 2)
        # Baseline line
        if len(self.clicks_baseline) == 1:
            cv2.circle(vis, tuple(map(int, self.clicks_baseline[0])), 5, (255, 0, 0), -1)
        elif len(self.clicks_baseline) == 2:
            cv2.line(vis, tuple(map(int, self.clicks_baseline[0])), tuple(map(int, self.clicks_baseline[1])), (255, 0, 0), 2)
        # Reference line
        if len(self.clicks_ref) == 1:
            cv2.circle(vis, tuple(map(int, self.clicks_ref[0])), 5, (0, 0, 255), -1)
        elif len(self.clicks_ref) == 2:
            cv2.line(vis, tuple(map(int, self.clicks_ref[0])), tuple(map(int, self.clicks_ref[1])), (0, 0, 255), 2)
        return to_rgb_uint8(vis), status

    def refresh_overlay(self, status: str = ""):
        if self.img_bgr is None:
            return gr.update(), "Load an image first."
        vis = self.img_bgr.copy()
        if self.label_top is not None and self.Hinv is not None:
            label_img, overlay = self._warp_labels_to_image()
            if overlay is not None:
                alpha = 0.45
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0.0)
        # Draw baseline/ref lines for context
        if len(self.clicks_baseline) == 2:
            cv2.line(vis, tuple(map(int, self.clicks_baseline[0])), tuple(map(int, self.clicks_baseline[1])), (255, 0, 0), 2)
        if len(self.clicks_ref) == 2:
            cv2.line(vis, tuple(map(int, self.clicks_ref[0])), tuple(map(int, self.clicks_ref[1])), (0, 0, 255), 2)
        return to_rgb_uint8(vis), status

    # ---------------
    # Export masks
    # ---------------
    def export(self, out_dir: str = "export"):
        if self.img_bgr is None or self.Hinv is None:
            return gr.update(), "Nothing to export."
        os.makedirs(out_dir, exist_ok=True)
        label_img, overlay = self._warp_labels_to_image()
        if label_img is None:
            return gr.update(), "No labels to export."
        exported = []
        for i, band in enumerate(self.bands, start=1):
            mask_bw = np.where(label_img == i, 255, 0).astype(np.uint8)
            if mask_bw.max() == 0:
                continue
            fn = os.path.join(out_dir, f"mask_{i:02d}_{band.label}.png")
            cv2.imwrite(fn, mask_bw)
            exported.append(fn)
        combined = np.where(label_img > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "mask_combined.png"), combined)
        overlay_rgb = to_rgb_uint8(overlay) if overlay is not None else None
        if overlay_rgb is not None:
            cv2.imwrite(os.path.join(out_dir, "overlay_preview.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        meta = {
            "image_path": self.img_path,
            "image_size": list(self.img_bgr.shape[:2][::-1]),
            "plane_quad_img_raw": None if self.quad_img_raw is None else self.quad_img_raw.tolist(),
            "plane_quad_img_extended": None if self.quad_img_extended is None else self.quad_img_extended.tolist(),
            "topdown_size": [self.top_w, self.top_h],
            "padding": [self.pad_x, self.pad_y],
            "meters_per_pixel": self.m_per_px,
            "baseline_img": self.clicks_baseline,
            "reference_img": self.clicks_ref,
            "reference_cm": self.ref_cm,
            "bands": [asdict(b) for b in self.bands],
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return gr.update(), f"Exported {len(exported)} masks + combined mask to '{out_dir}'."

    def set_invert_normal(self, flag: bool):
        self.invert_normal = bool(flag)
        return self.preview(status=f"Invert normal: {'ON' if self.invert_normal else 'OFF'}")


# ---------------------------
# Gradio UI
# ---------------------------
app = StreetBands()

def ui_load(img, img_info):
    path = getattr(img_info, 'name', None) if img_info is not None else None
    return app.load_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), path)


def ui_click(evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    return app.click_image(float(x), float(y))


def ui_start_over():
    return app.start_over()


def ui_add_band(label, from_cm, to_cm):
    return app.add_band(label or f"band{len(app.bands)+1}", from_cm, to_cm)


def ui_del_last():
    return app.delete_last_band()


def ui_clear():
    return app.clear_bands()


def ui_refresh():
    return app.refresh_overlay(status="Refreshed.")


def ui_export(out_dir):
    return app.export(out_dir)


def ui_invert_normal(flag):
    return app.set_invert_normal(flag)

with gr.Blocks(title="Street Bands v4.1 — bottom‑extended & click‑driven", analytics_enabled=False) as demo:
    gr.Markdown("## Street Bands v4.1 — bottom‑extended & click‑driven " \
    "Flow: 4 clicks plane → 2 clicks baseline → 2 clicks reference (length in cm) → add bands. " \
    "No image refresh for the first 3 plane clicks to keep things snappy.")

    with gr.Row():
        with gr.Column(scale=3):
            in_img = gr.Image(label="Input image (click)", interactive=True)
            in_img.upload(fn=ui_load, inputs=[in_img, in_img], outputs=[in_img, gr.Textbox(visible=False)])
            in_img.select(fn=ui_click, outputs=[in_img, gr.Textbox(visible=False)])

            with gr.Row():
                btn_reset = gr.Button("Start over")
                invert_toggle = gr.Checkbox(label="Invert normal (use right-hand side)", value=False)
            btn_reset.click(ui_start_over, outputs=[in_img, gr.Textbox(visible=False)])
            invert_toggle.change(ui_invert_normal, inputs=[invert_toggle], outputs=[in_img, gr.Textbox(visible=False)])

            gr.Markdown("### Bands (in centimeters)")
            with gr.Row():
                band_label = gr.Textbox(label="Label", value="lane")
                from_cm = gr.Number(label="from_cm", value=0)
                to_cm = gr.Number(label="to_cm", value=200)
            with gr.Row():
                add_btn = gr.Button("Add band")
                del_btn = gr.Button("Delete last band")
                clr_btn = gr.Button("Clear bands")
            add_btn.click(ui_add_band, inputs=[band_label, from_cm, to_cm], outputs=[in_img, gr.Textbox(visible=False)])
            del_btn.click(ui_del_last, outputs=[in_img, gr.Textbox(visible=False)])
            clr_btn.click(ui_clear, outputs=[in_img, gr.Textbox(visible=False)])

            gr.Markdown("### Reference length (cm) & Export")
            ref_cm_box = gr.Number(label="Reference length (cm)", value=100)
            ref_cm_box.change(lambda v: setattr(app, 'ref_cm', float(v)) or (gr.update(),), inputs=[ref_cm_box])

            with gr.Row():
                refresh_btn = gr.Button("Refresh overlay")
                out_dir = gr.Textbox(label="Export folder", value="export")
                export_btn = gr.Button("Export masks")
            refresh_btn.click(ui_refresh, outputs=[in_img, gr.Textbox(visible=False)])
            export_btn.click(ui_export, inputs=[out_dir], outputs=[gr.Image(visible=False), gr.Textbox()])

        with gr.Column(scale=2):
            gr.Markdown("### Status & Bands")
            band_table = gr.JSON(value={})
            def band_table_state():
                rows = []
                for i, b in enumerate(app.bands, start=1):
                    rows.append({"#": i, "label": b.label, "from_cm": b.from_cm, "to_cm": b.to_cm})
                return {"mode": app.mode, "meters_per_pixel": app.m_per_px, "bands": rows}
            refresh_side = gr.Button("Refresh band list")
            def refresh_side_action():
                return json.dumps(band_table_state(), indent=2)
            refresh_side.click(fn=refresh_side_action, outputs=[band_table])

    if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False, show_error=True)
