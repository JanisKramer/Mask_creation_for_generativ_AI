# Mask_creation_for_generativ_AI

Toolbox for creating **perspective-correct image masks** for streetscape edits (e.g., inpainting / image-to-image).
The repo contains two main approaches:

1. **Parametric mask creation** (metric bands on a road plane; e.g., lane/sidewalk areas) via a Gradio UI.
2. **DXF → image masks** by projecting 2D CAD geometry onto a street photo using a homography from matched control points.

This is meant to support workflows where edits must stay **geometrically plausible** (perspective consistent) and at least **roughly metric** (e.g., “a 1.5 m bike lane” is actually 1.5 m in the scene).

---

## Repository layout

- `Parametric_Mask_Creation/`  
  Gradio apps to define a road plane in the image and generate masks from metric/parametric inputs.
  Output examples are written to `Parametric_Mask_Creation/export/`.

- `DXF_2_Imagemasks/`  
  Scripts to project CAD geometry from DXF into the image and rasterize it into a binary mask.
  Includes example data in `DXF_2_Imagemasks/Untersiggenthal/`.

- `Reocuring_Elements/`  
  Prototypes for recurring objects (e.g., tree placement along the road plane) via a Gradio UI.

---

## Installation

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

#### Note on tkinter (file dialogs)
Some scripts optionally use `tkinter` for file pickers.
- Windows/macOS Python usually includes it.
- On Ubuntu/Debian you may need:
```bash
sudo apt-get install python3-tk
```

---

## Quickstart

### A) Parametric masks (road-plane “bands”) — Gradio UI

The most recent parametric script in this repo is:

```bash
python Parametric_Mask_Creation/251028_plane_negative_fix_v4_5.py
```

Then open the shown local URL (typically `http://127.0.0.1:7860`) in your browser.

Typical flow:
1. Load a street image.
2. Define the **road plane** in the image (4 corner points).
3. Provide a **metric reference** (e.g., a known length / street width) so the bands can be scaled.
4. Add band definitions (in cm; negatives allowed) to generate masks.
5. Export masks to `Parametric_Mask_Creation/export/` (combined + per-band masks + `meta.json`).

Outputs (example names):
- `mask_combined.png`
- `mask_01_<label>.png`, `mask_02_<label>.png`, ...
- `meta.json` (inputs and geometry metadata)

---

### B) Recurring elements (tree mask prototype) — Gradio UI

```bash
python Reocuring_Elements/2510131_reocuring_object_v3.py
```

Open the local Gradio URL (typically `http://127.0.0.1:7861`).

This tool focuses on fast placement of repeated objects (e.g., trees) along a street plane and exporting a corresponding mask.

---

### C) DXF → image mask (CAD projection)

Main script:
```bash
python DXF_2_Imagemasks/dxf2image_create_config.py --help
```

Minimal run (with file dialogs if you omit paths):
```bash
python DXF_2_Imagemasks/dxf2image_create_config.py
```

Typical run (explicit paths):
```bash
python DXF_2_Imagemasks/dxf2image_create_config.py \
  --image DXF_2_Imagemasks/Untersiggenthal/20251031_114405.jpg \
  --dxf   DXF_2_Imagemasks/Untersiggenthal/Us_version_1.dxf \
  --src-csv DXF_2_Imagemasks/Untersiggenthal/Us.csv \
  --out  DXF_2_Imagemasks/Untersiggenthal/Us_Maske_1.png \
  --save-overlay DXF_2_Imagemasks/Untersiggenthal/Us_Overlay_1.png
```

How it works (high level):
1. Provide **6 DXF coordinates** (from `--src-csv` or `--src-corners`).
2. Click the **same 6 points** in the image in the same order.
3. A homography is estimated (optionally with RANSAC).
4. CAD geometry is warped into the image and rasterized as a black/white mask.

Optional outputs:
- `--save-overlay` : mask visualized on top of the photo
- `--save-diagnostics` : per-point error vectors + diagnostics overlay
- `--save-config` : JSON config to reproduce the run (useful for batch processing)

> Note: `DXF_2_Imagemasks/dxf2image_load_config.py` expects helper functions from a `dxf2mask.py`
> located in the same folder. In this repo, the older implementation is in `DXF_2_Imagemasks/old/`.

---

## Common pitfalls (aka: what breaks this)

- **Planar assumption**: Homography/IPM assumes the edited surface is a plane (the road). Vertical objects (façades, cars) do not map correctly.
- **Occlusions**: If curbs/markings are blocked (parked cars), “metric” placement becomes uncertain.
- **Weak scale**: If you only use “street width” with noisy detection, scale drift happens. A known two-point distance is more reliable.
- **DXF layering**: If the DXF has multiple layers, use `--layer` to restrict which entities are rasterized.

---

## Suggested next steps (future work)

- Semi-automatic ground-plane scaffold via segmentation + minimal user correction, then template placement (mask as an internal artifact).
- Export both raster masks and vector primitives (DXF/SVG) for clean downstream editing.
- Add a small test set + timing benchmark (mask creation time vs. manual editing).

---

## License

No license file is included yet. Add a `LICENSE` if you plan to share or reuse this code outside personal/project work.
