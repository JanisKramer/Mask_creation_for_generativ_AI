# Street Bands — Parametric, Perspective-Correct Mask Generator (Straight Roads)

A fast line-first tool to create **perspective-correct bands** (road, cycle lane, sidewalk, green strip, etc.) from a single street photo.  
You define one ground plane once, then draw a **baseline + reference length** and type **offsets in cm** to get metric-true masks.

## Quick start (GUI)

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
python street_bands.py
```

1) **Upload an image.**  
2) **Ground plane (once):** click **4 points** in this order: **BL, TL, TR, BR** (bottom-left, top-left, top-right, bottom-right).  
   - The tool **extends the side edges down to the bottom** of the photo (top edge stays as clicked).  
   - Click **Lock plane**.  
3) **Baseline + Reference:**  
   - Click **2 baseline points** (e.g., along curb/road edge).  
   - Click **2 reference points** near the baseline (a known real-world length).  
   - Enter **reference length (cm)** and click **Lock baseline + reference**.  
4) **Add bands:** type **From (cm)** and **To (cm)** (e.g., `-90 → 0` for a 0.9 m cycle lane to the “left” of the baseline; `0 → 200` for a 2.0 m band to the “right`).  
   - Click **Add band** for each layer you need.  
   - Use **Delete last band** or **Clear all bands** as needed.  
5) **Export:** choose an output folder and click **Export**.

### Outputs
- `mask_<label>.png` — one 8-bit black/white mask per band (full-image size)  
- `overlay_planes.png` — color overlay preview for quick QA  
- `meta.json` — plane, baseline, scale (m/px), and full band definitions

## What makes masks “metric-true”

- A one-time **homography** rectifies the ground plane → consistent geometry.  
- Bands are built in the **top-down domain** by offsetting the baseline **by centimeters**, then warped back to the photo.  
- Negative ranges are supported (e.g., `-90 → 0`), and very thin bands are auto-clamped to at least ~1 px so they don’t disappear.

## Controls & tips

- **Click order matters** for the plane: **BL → TL → TR → BR**.  
- **Reference length** should be close to your baseline for best local scale.  
- For a visible first test, try a wider span (e.g., `0 → 200` cm), then refine.  
- Straight roads only in this version (baseline is a straight line).  
- Overlay is semi-transparent; masks are crisp (no alpha) for diffusion workflows.

## Requirements

`requirements.txt` (typical)
```
gradio
opencv-python
numpy
```

## Troubleshooting

- **Gradio / pandas option error** (`future.no_silent_downcasting`):  
  The app already disables analytics in code. If you still see it on some setups, set the env var before running:
  - Windows (CMD): `set GRADIO_ANALYTICS_ENABLED=false`
  - PowerShell: `$env:GRADIO_ANALYTICS_ENABLED="false"`
  - Or upgrade: `pip install -U gradio pandas`

- **Band not visible after “Add band”**  
  - Ensure you **locked baseline + reference** first.  
  - Try a **wider** `From/To` range to confirm visibility.  
  - Make sure you typed **centimeters** (not meters).

## Roadmap (next)

- “One-sided strips” that extend bands toward the camera for always-full coverage.  
- Curved baselines (polyline offsets) for corners.  
- Optional mask feathering and naming presets (e.g., `road`, `sidewalk`, `cycle`, `green`).

---

Happy mapping!
