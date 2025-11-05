#!/usr/bin/env python3
"""
dxf2mask_from_config.py

Use a previously saved JSON config (from dxf2mask.py --save-config)
to create a mask (and optional overlay/diagnostics) WITHOUT clicking
points or selecting the source CSV again.

You pick:
- image (photo)
- DXF
- JSON config
- output mask

The config file structure (version 1):

{
  "version": 1,
  "src_points": [[x,y], ... 6],
  "image_points": [[x,y], ... 6],
  "layer": null or "layer_name",
  "tolerance": 0.001,
  "use_ransac": false,
  "ransac_thresh": 4.0
}
"""

import argparse
import json
from typing import Optional

import cv2
import ezdxf
import numpy as np

# Try to reuse the helper functions from dxf2mask.py
try:
    from dxf2mask import (
        _largest_closed_poly_from_msp,
        _reprojection_errors,
        _print_stats,
        _draw_diagnostics,
    )
except ImportError as e:
    raise ImportError(
        "Could not import helpers from dxf2mask.py. "
        "Make sure dxf2mask.py is in the same folder and named exactly 'dxf2mask.py'."
    ) from e

# For interactive file selection (same style as dxf2mask.py)
try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_OK = True
except Exception:
    _TK_OK = False


def run_from_config(
    image_path: str,
    dxf_path: str,
    config_path: str,
    out_mask_path: str,
    *,
    save_diag: Optional[str] = None,
    save_overlay: Optional[str] = None,
) -> None:
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    print(f"Loaded image {image_path} with size {w} x {h}")

    # Load DXF polygon
    print(f"Reading DXF file: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Load config JSON
    print(f"Reading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    version = cfg.get("version", 1)
    if version != 1:
        print(f"Warning: config version {version} not explicitly supported (expected 1). Proceeding anyway.")

    src_points = np.array(cfg["src_points"], dtype=np.float32)
    dst_points = np.array(cfg["image_points"], dtype=np.float32)

    if src_points.shape != (6, 2) or dst_points.shape != (6, 2):
        raise ValueError(
            f"Config file must contain 6 src_points and 6 image_points of shape (6,2), "
            f"got {src_points.shape} and {dst_points.shape}."
        )

    layer = cfg.get("layer", None)
    tol = float(cfg.get("tolerance", 1e-3))
    use_ransac = bool(cfg.get("use_ransac", False))
    ransac_thresh = float(cfg.get("ransac_thresh", 4.0))

    print("Config parameters:")
    print(f"  layer          = {layer!r}")
    print(f"  tolerance      = {tol}")
    print(f"  use_ransac     = {use_ransac}")
    print(f"  ransac_thresh  = {ransac_thresh}")

    # Get polygon from DXF
    poly = _largest_closed_poly_from_msp(msp, layer=layer, line_snap_tol=tol)

    # Homography from config points
    method = cv2.RANSAC if use_ransac else 0
    H, mask_inliers = cv2.findHomography(
        src_points,
        dst_points,
        method=method,
        ransacReprojThreshold=ransac_thresh,
    )
    if H is None:
        raise RuntimeError("Homography computation failed.")

    # Stats
    dists, proj = _reprojection_errors(H, src_points, dst_points)
    _print_stats(dists)

    # Warp polygon & make mask
    poly_h = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), H).reshape(-1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly_h).astype(np.int32)], 255)
    ok = cv2.imwrite(out_mask_path, mask)
    if not ok:
        raise IOError(f"Could not write mask to {out_mask_path}")
    print(f"Saved mask: {out_mask_path}")

    # Overlay
    if save_overlay:
        overlay = img.copy()
        alpha = 0.35
        mask_bool = mask > 0
        if np.any(mask_bool):
            overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * 255).astype(np.uint8)
        ok2 = cv2.imwrite(save_overlay, overlay)
        if ok2:
            print(f"Saved overlay: {save_overlay}")
        else:
            print("Warning: failed to save overlay image")

    # Diagnostics (error vectors, warped polygon)
    if save_diag:
        diag = _draw_diagnostics(img, dst_points, proj, poly_h)
        ok3 = cv2.imwrite(save_diag, diag)
        if ok3:
            print(f"Saved diagnostics: {save_diag}")
        else:
            print("Warning: failed to save diagnostics image")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DXF→Image mask using a saved JSON config (no clicking / no src CSV).",
    )
    parser.add_argument("--image", required=False, help="Path to street image (JPG/PNG).")
    parser.add_argument("--dxf", required=False, help="Path to DXF file.")
    parser.add_argument("--config", required=False, help="Path to config JSON exported by dxf2mask.py.")
    parser.add_argument("--out", required=False, help="Path to save the black/white mask PNG.")
    parser.add_argument("--save-diagnostics", default=None, help="PNG path for the error visualization overlay.")
    parser.add_argument("--save-overlay", default=None, help="PNG path for the mask-over-photo visual control.")
    args = parser.parse_args()

    # If something is missing, open file dialogs (VS Code friendly)
    if not (args.image and args.dxf and args.config and args.out and args.save_overlay):
        if not _TK_OK:
            raise SystemExit(
                "tkinter not available. Please provide --image, --dxf, --config, --out and optionally "
                "--save-overlay/--save-diagnostics on the command line."
            )

        root = tk.Tk()
        root.withdraw()

        if not args.image:
            args.image = filedialog.askopenfilename(
                title="Select street image",
                filetypes=[("Images", ".png .jpg .jpeg .bmp .tif .tiff"), ("All files", "*.*")],
            )

        if not args.dxf:
            args.dxf = filedialog.askopenfilename(
                title="Select DXF file",
                filetypes=[("DXF", ".dxf"), ("All files", "*.*")],
            )

        if not args.config:
            args.config = filedialog.askopenfilename(
                title="Select JSON config file",
                filetypes=[("Config", ".json .txt"), ("All files", "*.*")],
            )

        if not args.out:
            args.out = filedialog.asksaveasfilename(
                title="Save output MASK as",
                defaultextension=".png",
                filetypes=[("PNG", ".png"), ("All files", "*.*")],
            )

        # Ask for overlay path (optional; cancel to skip)
        if args.save_overlay is None:
            import os
            suggested = None
            try:
                if args.out:
                    base, _ = os.path.splitext(args.out)
                    suggested = base + "_overlay.png"
            except Exception:
                suggested = None
            args.save_overlay = filedialog.asksaveasfilename(
                title="Save OVERLAY (mask on photo) as — Cancel to skip",
                initialfile=(suggested if suggested else "overlay.png"),
                defaultextension=".png",
                filetypes=[("PNG", ".png"), ("All files", "*.*")],
            )
            if not args.save_overlay:
                args.save_overlay = None

        # Ask for diagnostics path (optional; cancel to skip)
        if args.save_diagnostics is None:
            import os
            suggested_diag = None
            try:
                if args.out:
                    base, _ = os.path.splitext(args.out)
                    suggested_diag = base + "_diag.png"
            except Exception:
                suggested_diag = None
            args.save_diagnostics = filedialog.asksaveasfilename(
                title="Save DIAGNOSTICS image as — Cancel to skip",
                initialfile=(suggested_diag if suggested_diag else "diagnostics.png"),
                defaultextension=".png",
                filetypes=[("PNG", ".png"), ("All files", "*.*")],
            )
            if not args.save_diagnostics:
                args.save_diagnostics = None

        root.destroy()

    if not args.image or not args.dxf or not args.config or not args.out:
        raise SystemExit("Image, DXF, config, and output paths are required.")

    run_from_config(
        image_path=args.image,
        dxf_path=args.dxf,
        config_path=args.config,
        out_mask_path=args.out,
        save_diag=args.save_diagnostics,
        save_overlay=args.save_overlay,
    )


if __name__ == "__main__":
    main()
