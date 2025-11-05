#!/usr/bin/env python3
"""
crop_512_same_folder.py
-----------------------
Simple script to crop an image named 'input.png' (in the same folder as this file)
to a square using a chosen gravity, then resize to exactly 512x512 and save as 'output.png'.

Edit the variables in the CONFIG section below to change behavior.
"""

# ======================
# CONFIG (edit these)
# ======================
INPUT_FILENAME   = "input.png"        # The input image in the same folder as this script
OUTPUT_FILENAME  = "output.png"       # Output image name
TARGET_SIZE      = (512, 512)         # Final size (width, height)
GRAVITY          = "center"           # One of: center, top, bottom, left, right
JPEG_QUALITY     = 95                 # If saving .jpg/.jpeg
# ======================

from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
import sys

GRAVITIES = {"center", "top", "bottom", "left", "right"}


def square_crop_box(w: int, h: int, gravity: str) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) crop box for a 1:1 crop using gravity."""
    if gravity not in GRAVITIES:
        raise ValueError(f"Invalid gravity '{gravity}'. Choose from: {sorted(GRAVITIES)}")

    if w == h:
        return (0, 0, w, h)

    if w > h:
        # Wider than tall: crop width
        side = h
        if gravity == "left":
            left = 0
        elif gravity == "right":
            left = w - side
        else:  # center/top/bottom behave the same for width crop
            left = (w - side) // 2
        top = 0
    else:
        # Taller than wide: crop height
        side = w
        left = 0
        if gravity == "top":
            top = 0
        elif gravity == "bottom":
            top = h - side
        else:  # center/left/right behave the same for height crop
            top = (h - side) // 2

    return (left, top, left + side, top + side)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    in_path = script_dir / INPUT_FILENAME
    out_path = script_dir / OUTPUT_FILENAME

    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with Image.open(in_path) as im:
            # Respect EXIF orientation
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            box = square_crop_box(w, h, GRAVITY)
            im = im.crop(box)
            im = im.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            suffix = out_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGB")
                im.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            else:
                im.save(out_path)

        print(f"[OK] {in_path.name} -> {out_path.name} ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}, gravity={GRAVITY})")
    except Exception as e:
        print(f"[FAIL] {in_path.name}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
