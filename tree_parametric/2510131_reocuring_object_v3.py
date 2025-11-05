#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree placer (ground plane) – USES tree.png SHAPE
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import gradio as gr
import numpy as np

# ----------------------------------------------------
# config
# ----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SPRITE_PATH = SCRIPT_DIR / "tree.png"   # your file

EPS = 1e-6


# ----------------------------------------------------
# helpers
# ----------------------------------------------------
def to_rgb(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    def load_tree_mask(path: Path) -> Optional[np.ndarray]:
        """
        Load tree.png and return a 0/255 mask.
        If the PNG has an alpha channel but it's fully opaque (all 255) or fully transparent (all 0),
        we ignore that alpha and instead threshold the RGB.
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] could not load {path}")
            return None

        # RGBA
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]

            # is alpha uniform?
            amin = int(alpha.min())
            amax = int(alpha.max())
            if amin == amax:
                # alpha is useless -> use RGB
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                return mask
            else:
                # real alpha
                mask = (alpha > 20).astype(np.uint8) * 255
                return mask

        # RGB
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            return mask

        # GRAY
        if img.ndim == 2:
            _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
            return mask

        return None



def order_quad_bl_tl_tr_br(pts: np.ndarray) -> np.ndarray:
    idx = np.argsort(pts[:, 1])
    top2 = pts[idx[:2]]
    bot2 = pts[idx[2:]]
    tl, tr = sorted(top2, key=lambda p: p[0])
    bl, br = sorted(bot2, key=lambda p: p[0])
    return np.array([bl, tl, tr, br], dtype=np.float32)


def compute_homography_from_plane(quad_img: np.ndarray, real_width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    bl, tl, tr, br = quad_img
    img_w_bottom = np.linalg.norm(br - bl)
    img_h_left   = np.linalg.norm(tl - bl)
    img_h_right  = np.linalg.norm(tr - br)
    img_h = 0.5 * (img_h_left + img_h_right)

    ground_w = real_width_m
    aspect = img_h / (img_w_bottom + EPS)
    ground_h = ground_w * aspect

    pts_img = np.array([bl, tl, tr, br], dtype=np.float32)
    pts_gnd = np.array([
        [0.0,        0.0],
        [0.0,        ground_h],
        [ground_w,   ground_h],
        [ground_w,   0.0],
    ], dtype=np.float32)

    H     = cv2.getPerspectiveTransform(pts_img, pts_gnd)
    H_inv = cv2.getPerspectiveTransform(pts_gnd, pts_img)
    return H, H_inv


class GroundTreeApp:
    def __init__(self, sprite_path: Path):
        self.tree_mask = load_tree_mask(sprite_path)
        self.has_tree  = self.tree_mask is not None

        self.img_bgr: Optional[np.ndarray] = None

        # clicks
        self.clicks_plane: List[Tuple[float, float]] = []
        self.clicks_v1: List[Tuple[float, float]] = []
        self.clicks_v2: List[Tuple[float, float]] = []
        self.clicks_line: List[Tuple[float, float]] = []

        # homographies
        self.H_img2gnd: Optional[np.ndarray] = None
        self.H_gnd2img: Optional[np.ndarray] = None

        # params
        self.plane_width_m = 10.0
        self.tree_height_m = 3.0
        self.spacing_m     = 10.0
        self.use_spacing   = True
        self.n_trees       = 4

        self.mode = "plane"

    def set_image(self, img_bgr):
        # keep the loaded mask
        m = self.tree_mask
        self.__init__(SPRITE_PATH)
        self.tree_mask = m
        self.has_tree  = m is not None
        self.img_bgr   = img_bgr

    # ------------- vertical scale from two rods (CLAMP after far) -------------
    def _vertical_pixels_per_meter_at_gnd(self, gnd_pt: np.ndarray) -> float:
        if self.H_img2gnd is None or len(self.clicks_v1) < 2:
            return 100.0

        # near
        v1_base_img = np.array(self.clicks_v1[0], dtype=np.float32)[None, None, :]
        v1_base_gnd = cv2.perspectiveTransform(v1_base_img, self.H_img2gnd)[0, 0]
        v1_top_img  = np.array(self.clicks_v1[1], dtype=np.float32)
        px1         = float(np.linalg.norm(v1_top_img - self.clicks_v1[0]))

        # if no far → constant
        if len(self.clicks_v2) < 2:
            return px1

        # far
        v2_base_img = np.array(self.clicks_v2[0], dtype=np.float32)[None, None, :]
        v2_base_gnd = cv2.perspectiveTransform(v2_base_img, self.H_img2gnd)[0, 0]
        v2_top_img  = np.array(self.clicks_v2[1], dtype=np.float32)
        px2         = float(np.linalg.norm(v2_top_img - self.clicks_v2[0]))

        y1 = v1_base_gnd[1]
        y2 = v2_base_gnd[1]
        yg = gnd_pt[1]

        if abs(y2 - y1) < 1e-4:
            return px1

        if yg <= y1:
            return px1 * 1.25

        if yg <= y2:
            alpha = (yg - y1) / (y2 - y1)
            v = px1 + alpha * (px2 - px1)
            return max(2.0, v)

        # AFTER FAR -> clamp (this fixed your “tiny last”)
        return max(2.0, px2)

    # ------------- tree positions in ground -------------
    def _tree_positions_on_ground(self) -> List[np.ndarray]:
        if self.H_img2gnd is None or len(self.clicks_line) < 2:
            return []

        p0_img = np.array(self.clicks_line[0], dtype=np.float32)[None, None, :]
        p1_img = np.array(self.clicks_line[1], dtype=np.float32)[None, None, :]
        p0_gnd = cv2.perspectiveTransform(p0_img, self.H_img2gnd)[0, 0]
        p1_gnd = cv2.perspectiveTransform(p1_img, self.H_img2gnd)[0, 0]

        dir_g = p1_gnd - p0_gnd
        dist  = float(np.linalg.norm(dir_g))
        if dist < 1e-6:
            return []
        dir_g /= dist

        pts = []
        if self.use_spacing:
            m = 0.0
            while m <= dist + 1e-4:
                pts.append(p0_gnd + dir_g * m)
                m += self.spacing_m
        else:
            for i in range(self.n_trees):
                t = i / max(1, (self.n_trees - 1))
                pts.append(p0_gnd + dir_g * dist * t)
        return pts

    # ------------- preview -------------
    def preview(self):
        vis = self.img_bgr.copy()

        # plane
        for p in self.clicks_plane:
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
        if len(self.clicks_plane) == 4:
            quad = order_quad_bl_tl_tr_br(np.array(self.clicks_plane, dtype=np.float32))
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 2)

        # near / far
        if len(self.clicks_v1) >= 1:
            cv2.circle(vis, tuple(map(int, self.clicks_v1[0])), 5, (0, 0, 255), -1)
        if len(self.clicks_v1) == 2:
            cv2.line(vis,
                     tuple(map(int, self.clicks_v1[0])),
                     tuple(map(int, self.clicks_v1[1])),
                     (0, 0, 255), 2)

        if len(self.clicks_v2) >= 1:
            cv2.circle(vis, tuple(map(int, self.clicks_v2[0])), 5, (255, 0, 0), -1)
        if len(self.clicks_v2) == 2:
            cv2.line(vis,
                     tuple(map(int, self.clicks_v2[0])),
                     tuple(map(int, self.clicks_v2[1])),
                     (255, 0, 0), 2)

        # line
        if len(self.clicks_line) >= 1:
            cv2.circle(vis, tuple(map(int, self.clicks_line[0])), 5, (0, 255, 0), -1)
        if len(self.clicks_line) == 2:
            cv2.line(vis,
                     tuple(map(int, self.clicks_line[0])),
                     tuple(map(int, self.clicks_line[1])),
                     (0, 255, 0), 2)

        # planned trees
        if self.H_img2gnd is not None and len(self.clicks_line) == 2 and self.H_gnd2img is not None:
            gnd_pts = self._tree_positions_on_ground()
            if gnd_pts:
                gnd_arr = np.array(gnd_pts, dtype=np.float32)[None, :, :]
                img_pts = cv2.perspectiveTransform(gnd_arr, self.H_gnd2img)[0]
                for p in img_pts:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

        return vis

    # ------------- build mask -------------
    def build_mask(self):
        h, w = self.img_bgr.shape[:2]
        out_mask = np.zeros((h, w), dtype=np.uint8)

        if not self.has_tree:
            return out_mask, 0

        gnd_pts = self._tree_positions_on_ground()
        if not gnd_pts:
            return out_mask, 0

        gnd_arr = np.array(gnd_pts, dtype=np.float32)[None, :, :]
        img_pts = cv2.perspectiveTransform(gnd_arr, self.H_gnd2img)[0]

        stamped = 0
        base_mask = self.tree_mask  # HxW

        for gnd_pt, img_pt in zip(gnd_pts, img_pts):
            vppm = self._vertical_pixels_per_meter_at_gnd(gnd_pt)
            tree_px_h = int(round(vppm * self.tree_height_m))
            tree_px_h = max(2, tree_px_h)

            sh, sw = base_mask.shape
            scale  = tree_px_h / float(sh)
            new_w  = max(2, int(round(sw * scale)))
            new_h  = max(2, int(round(sh * scale)))

            mask_s = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            x = int(round(img_pt[0]))
            y = int(round(img_pt[1]))
            tl_x = x - new_w // 2
            tl_y = y - new_h

            x0 = max(0, tl_x); y0 = max(0, tl_y)
            x1 = min(w, tl_x + new_w); y1 = min(h, tl_y + new_h)
            if x0 >= x1 or y0 >= y1:
                continue
            sx0 = x0 - tl_x; sy0 = y0 - tl_y
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)

            sub = mask_s[sy0:sy1, sx0:sx1]
            out_mask[y0:y1, x0:x1] = np.where(sub >= 128, 255, out_mask[y0:y1, x0:x1])
            stamped += 1

        return out_mask, stamped


# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------
app = GroundTreeApp(SPRITE_PATH)


def ui_load(img, info):
    app.set_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return to_rgb(app.preview()), "1/5: click 4 ground points."


def ui_click(evt: gr.SelectData):
    x, y = float(evt.index[0]), float(evt.index[1])

    if app.mode == "plane":
        app.clicks_plane.append((x, y))
        if len(app.clicks_plane) == 4:
            quad = order_quad_bl_tl_tr_br(np.array(app.clicks_plane, dtype=np.float32))
            H, Hinv = compute_homography_from_plane(quad, app.plane_width_m)
            app.H_img2gnd = H
            app.H_gnd2img = Hinv
            app.mode = "v1"
            return to_rgb(app.preview()), "2/5: near vertical (base, top)."
        return to_rgb(app.preview()), f"1/5: plane {len(app.clicks_plane)}/4…"

    if app.mode == "v1":
        app.clicks_v1.append((x, y))
        if len(app.clicks_v1) == 2:
            app.mode = "v2"
            return to_rgb(app.preview()), "3/5: far vertical (base, top)."
        return to_rgb(app.preview()), "2/5: near vertical 1/2…"

    if app.mode == "v2":
        app.clicks_v2.append((x, y))
        if len(app.clicks_v2) == 2:
            app.mode = "line"
            return to_rgb(app.preview()), "4/5: tree line (start, end)."
        return to_rgb(app.preview()), "3/5: far vertical 1/2…"

    if app.mode == "line":
        app.clicks_line.append((x, y))
        if len(app.clicks_line) == 2:
            app.mode = "ready"
            return to_rgb(app.preview()), "5/5: set spacing / N and export."
        return to_rgb(app.preview()), "4/5: tree line 1/2…"

    return to_rgb(app.preview()), "Ready."


def ui_params(plane_w, tree_h, spacing_m, n_trees, use_spacing):
    app.plane_width_m = float(plane_w)
    app.tree_height_m = float(tree_h)
    app.spacing_m     = float(spacing_m)
    app.n_trees       = int(n_trees)
    app.use_spacing   = bool(use_spacing)

    if len(app.clicks_plane) == 4 and app.img_bgr is not None:
        quad = order_quad_bl_tl_tr_br(np.array(app.clicks_plane, dtype=np.float32))
        H, Hinv = compute_homography_from_plane(quad, app.plane_width_m)
        app.H_img2gnd = H
        app.H_gnd2img = Hinv

    return to_rgb(app.preview()), "Parameters updated."


def ui_export(out_dir):
    if app.img_bgr is None:
        return None, "Need an image."
    if app.H_img2gnd is None or len(app.clicks_line) < 2 or len(app.clicks_v1) < 2:
        return None, "Need: plane(4), near(2), tree line(2)."
    os.makedirs(out_dir, exist_ok=True)
    mask, n = app.build_mask()
    out_path = os.path.join(out_dir, "trees_mask.png")
    cv2.imwrite(out_path, mask)
    return to_rgb(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), f"Saved {out_path} with {n} trees."


with gr.Blocks(title="Ground-plane tree placer (real tree.png)") as demo:
    gr.Markdown("Flow: 1) 4× ground 2) near 1m 3) far 1m 4) tree line 5) spacing / export")

    img_in = gr.Image(label="Image", interactive=True)
    status = gr.Markdown()

    img_in.upload(fn=ui_load, inputs=[img_in, img_in], outputs=[img_in, status])
    img_in.select(fn=ui_click, outputs=[img_in, status])

    with gr.Row():
        plane_w    = gr.Number(label="Plane bottom width (m)", value=10.0)
        tree_h     = gr.Number(label="Tree height (m)", value=3.0)
        spacing_m  = gr.Number(label="Spacing (m)", value=10.0)
        n_trees    = gr.Number(label="N trees (if not spacing)", value=4)
        use_spacing = gr.Checkbox(label="Use spacing (m)?", value=True)

    gr.Button("Apply params").click(
        fn=ui_params,
        inputs=[plane_w, tree_h, spacing_m, n_trees, use_spacing],
        outputs=[img_in, status]
    )

    out_dir = gr.Textbox(label="Export folder", value="export_trees_masks")
    gr.Button("Export mask").click(fn=ui_export, inputs=[out_dir], outputs=[img_in, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_api=False, show_error=True)
