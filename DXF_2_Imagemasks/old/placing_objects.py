#!/usr/bin/env python3
import os
import json
import csv
from typing import List, Tuple, Union

import cv2
import numpy as np
import gradio as gr


def _ensure_filepath(maybe_path: Union[str, dict, None], label: str) -> str:
    if maybe_path is None:
        raise ValueError(f"No {label} provided.")
    if isinstance(maybe_path, str):
        return maybe_path
    if isinstance(maybe_path, dict) and "name" in maybe_path:
        return maybe_path["name"]
    raise ValueError(f"Unexpected {label} object: {type(maybe_path)}")


def read_tree_points(csv_path: str) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    with open(csv_path, "r", newline="") as f:
        data = f.read()
        f.seek(0)
        sniff_delims = ",;\t "
        try:
            dialect = csv.Sniffer().sniff(data, delimiters=sniff_delims)
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
                # header or trash – skip
                continue

    if not pts:
        raise ValueError(f"No valid coordinates in {csv_path}")
    return np.array(pts, dtype=np.float32)


def load_homography_from_config(config_path: str) -> np.ndarray:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    src_points = np.array(cfg["src_points"], dtype=np.float32)
    dst_points = np.array(cfg["image_points"], dtype=np.float32)

    if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
        raise ValueError("Config must contain at least 4 src_points and 4 image_points.")

    use_ransac = bool(cfg.get("use_ransac", False))
    ransac_thresh = float(cfg.get("ransac_thresh", 4.0))
    method = cv2.RANSAC if use_ransac else 0

    H, inliers = cv2.findHomography(
        src_points, dst_points, method=method, ransacReprojThreshold=ransac_thresh
    )
    if H is None:
        raise RuntimeError("Could not compute homography from config.")
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return proj


def local_pixels_per_meter(H: np.ndarray, xy: Tuple[float, float], step_m: float = 1.0) -> float:
    x, y = xy
    p = np.array([[x, y]], dtype=np.float32)
    p2 = np.array([[x, y + step_m]], dtype=np.float32)
    q = apply_homography(H, p)[0]
    q2 = apply_homography(H, p2)[0]
    dist = np.hypot(q2[0] - q[0], q2[1] - q[1])
    return float(dist) / step_m if step_m != 0 else 0.0


def place_tree_mask(
    base_img_bgr: np.ndarray,
    tree_gray: np.ndarray,
    H: np.ndarray,
    tree_points_dxf: np.ndarray,
    tree_height_m: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    h, w = base_img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    overlay = base_img_bgr.copy()

    th0, tw0 = tree_gray.shape[:2]
    placed = 0

    for (X, Y) in tree_points_dxf:
        img_pt = apply_homography(H, np.array([[X, Y]], dtype=np.float32))[0]
        u, v = float(img_pt[0]), float(img_pt[1])

        if not (np.isfinite(u) and np.isfinite(v)):
            continue

        ppm = local_pixels_per_meter(H, (X, Y), step_m=1.0)
        if ppm <= 0:
            continue

        tree_h_px = int(round(ppm * tree_height_m))
        if tree_h_px < 3:
            continue

        scale = tree_h_px / float(th0)
        tree_w_px = int(round(tw0 * scale))
        if tree_w_px < 1:
            continue

        tree_resized = cv2.resize(tree_gray, (tree_w_px, tree_h_px), interpolation=cv2.INTER_AREA)

        # bottom-center binding
        x0 = int(round(u - tree_w_px / 2))
        y1 = int(round(v))
        y0 = y1 - tree_h_px

        if x0 >= w or y0 >= h or x0 + tree_w_px <= 0 or y1 <= 0:
            continue

        x0_clip = max(0, x0)
        y0_clip = max(0, y0)
        x1_clip = min(w, x0 + tree_w_px)
        y1_clip = min(h, y1)

        tx0 = x0_clip - x0
        ty0 = y0_clip - y0
        tx1 = tx0 + (x1_clip - x0_clip)
        ty1 = ty0 + (y1_clip - y0_clip)

        tree_crop = tree_resized[ty0:ty1, tx0:tx1]
        tree_bool = tree_crop > 0

        mask_region = mask[y0_clip:y1_clip, x0_clip:x1_clip]
        mask_region[tree_bool] = 255
        mask[y0_clip:y1_clip, x0_clip:x1_clip] = mask_region

        ov_region = overlay[y0_clip:y1_clip, x0_clip:x1_clip]
        ov_region[tree_bool] = (255, 255, 255)
        overlay[y0_clip:y1_clip, x0_clip:x1_clip] = ov_region

        placed += 1

    return mask, overlay, placed


def run_tree_placement(
    street_img_rgb,
    config_file,
    tree_img_rgba,
    tree_height_m,
    csv_file,
    out_dir,
    base_name,
):
    try:
        if street_img_rgb is None:
            return None, None, "No street image."

        config_path = _ensure_filepath(config_file, "config JSON")
        csv_path = _ensure_filepath(csv_file, "CSV file")

        if tree_img_rgba is None:
            return None, None, "No tree image."

        if not base_name:
            base_name = "trees"
        out_dir = (out_dir or "").strip() or "trees_export"
        os.makedirs(out_dir, exist_ok=True)

        # ---- Street image -> BGR ----
        street = street_img_rgb
        if street.ndim == 2:
            base_bgr = cv2.cvtColor(street, cv2.COLOR_GRAY2BGR)
        else:
            c = street.shape[2]
            if c == 3:
                base_bgr = cv2.cvtColor(street, cv2.COLOR_RGB2BGR)
            elif c == 4:
                base_bgr = cv2.cvtColor(street, cv2.COLOR_RGBA2BGR)
            else:
                # fallback: keep first 3 channels and hope for the best
                base_bgr = cv2.cvtColor(street[:, :, :3], cv2.COLOR_RGB2BGR)

        # ---- Tree image -> single-channel mask ----
        tree = tree_img_rgba
        if tree.ndim == 2:
            tree_gray = tree.astype(np.uint8)
        else:
            c = tree.shape[2]
            if c == 1:
                tree_gray = tree[:, :, 0]
            elif c == 3:
                tree_gray = cv2.cvtColor(tree, cv2.COLOR_RGB2GRAY)
            elif c == 4:
                # use alpha if it exists and is not trivial, because your PNG is white+alpha
                alpha = tree[:, :, 3]
                if np.any(alpha != alpha[0, 0]):
                    tree_gray = alpha
                else:
                    tree_gray = cv2.cvtColor(tree[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                tree_gray = tree[:, :, 0]

        tree_gray = tree_gray.astype(np.uint8)

        # Homography + points
        H = load_homography_from_config(config_path)
        tree_pts = read_tree_points(csv_path)

        mask, overlay_bgr, placed = place_tree_mask(
            base_bgr, tree_gray, H, tree_pts, float(tree_height_m)
        )

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")

        ok_mask = cv2.imwrite(mask_path, mask)
        ok_overlay = cv2.imwrite(overlay_path, overlay_bgr)

        status_lines = [
            f"Requested positions: {len(tree_pts)}",
            f"Trees placed (inside image): {placed}",
            f"Output folder: {os.path.abspath(out_dir)}",
        ]
        if ok_mask:
            status_lines.append(f"Mask saved as: {mask_path}")
        else:
            status_lines.append(f"ERROR: Could not save mask to {mask_path}")
        if ok_overlay:
            status_lines.append(f"Overlay saved as: {overlay_path}")
        else:
            status_lines.append(f"ERROR: Could not save overlay to {overlay_path}")

        return mask, overlay_rgb, "\n".join(status_lines)

    except Exception as e:
        # brutal honesty, as requested
        return None, None, f"Error: {e}"


with gr.Blocks(title="Tree placer from DXF config") as demo:
    gr.Markdown(
        "## Tree placer\n\n"
        "- Uses your DXF→image JSON config to get the homography\n"
        "- Places `tree.png` at DXF coordinates from a CSV\n"
        "- Scales each tree by local perspective (pixels-per-meter)\n"
    )

    with gr.Row():
        with gr.Column(scale=2):
            street_img = gr.Image(
                label="Street image",
                type="numpy",
            )
            tree_img = gr.Image(
                label="Tree mask / sprite (white tree on black, RGBA is fine)",
                type="numpy",
            )

            config_file = gr.File(
                label="DXF→image config JSON",
                file_types=[".json"],
            )
            csv_file = gr.File(
                label="Tree positions CSV (x,y per row)",
                file_types=[".csv", ".txt"],
            )

            tree_height = gr.Number(
                label="Tree height [meters]",
                value=4.0,
                precision=1,
            )
            out_dir = gr.Textbox(
                label="Output folder",
                value="trees_export",
            )
            base_name = gr.Textbox(
                label="Output base name",
                value="trees",
            )

            run_btn = gr.Button("Generate tree mask & overlay")

        with gr.Column(scale=2):
            mask_out = gr.Image(
                label="Tree mask (for compositing)",
                interactive=False,
            )
            overlay_out = gr.Image(
                label="Overlay on street image (preview)",
                interactive=False,
            )
            status_box = gr.Textbox(
                label="Status / log",
                interactive=False,
                lines=8,
            )

    run_btn.click(
        fn=run_tree_placement,
        inputs=[street_img, config_file, tree_img, tree_height, csv_file, out_dir, base_name],
        outputs=[mask_out, overlay_out, status_box],
    )


if __name__ == "__main__":
    demo.launch()
