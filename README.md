# Mask_creation_for_generativ_AI

Toolbox for creating **perspective-correct binary masks** for street-scene image editing.

**Main focus:** `DXF_2_Imagemasks/` — project **2D CAD/DXF geometry** into a street photo (homography) and export a mask for inpainting / generative pipelines.

> This README was created with the help of ChatGPT.

---

## What the DXF workflow does

You provide:
- a street photo
- a DXF (typically a polygon / outline you want as a mask)
- **6 control points** in DXF coordinates (CSV)
- and you click the **same 6 points** in the image (same order)

The script estimates a homography (optionally with RANSAC), warps the DXF geometry into the image, and writes:
- `mask.png` (black/white)
- `overlay.png` (mask visualized on top of the photo)
- optionally `*_diag.png` (reprojection error vectors)
- optionally `*_config.json` (so you can re-run without clicking)

### Hard limitation (don’t ignore this)
This assumes the edited surface is **planar** (road plane). It works for markings/areas on the ground. It will not correctly map vertical objects (facades, poles, cars).

---

## Install (Windows)

### 1) Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bat
pip install -r requirements.txt
```

---

## Quickstart (DXF → mask)

From the repo root:

### Option A — Run with file dialogs (fast)

```bat
python DXF_2_Imagemasks\dxf2image_create_config.py
```

You will be asked to pick:
- image
- DXF
- output mask path
- (optional) output overlay path
- (optional) output config path
- CSV/TXT file with the 6 DXF control points

Then you click 6 points in the image.

### Option B — Run with explicit paths (reproducible)

Example data is included in `DXF_2_Imagemasks/Untersiggenthal/`.

```bat
python DXF_2_Imagemasks\dxf2image_create_config.py ^
  --image DXF_2_Imagemasks\Untersiggenthal\20251031_114405.jpg ^
  --dxf   DXF_2_Imagemasks\Untersiggenthal\Us_version_1.dxf ^
  --src-csv DXF_2_Imagemasks\Untersiggenthal\Us.csv ^
  --out  DXF_2_Imagemasks\Untersiggenthal\mask.png ^
  --save-overlay DXF_2_Imagemasks\Untersiggenthal\overlay.png ^
  --save-config  DXF_2_Imagemasks\Untersiggenthal\config.json ^
  --save-diagnostics DXF_2_Imagemasks\Untersiggenthal\diag.png
```

---

## Re-run without clicking (use the saved config)

If you saved a `config.json`, you can generate the mask again without clicking points:

```bat
python DXF_2_Imagemasks\dxf2image_load_config.py ^
  --image DXF_2_Imagemasks\Untersiggenthal\20251031_114405.jpg ^
  --dxf   DXF_2_Imagemasks\Untersiggenthal\Us_version_1.dxf ^
  --config DXF_2_Imagemasks\Untersiggenthal\config.json ^
  --out DXF_2_Imagemasks\Untersiggenthal\mask_from_config.png ^
  --save-overlay DXF_2_Imagemasks\Untersiggenthal\overlay_from_config.png
```

---

## Input format: the 6-point CSV

`--src-csv` must contain **exactly 6 rows** of `x,y` in DXF units.
- Header is allowed.
- Comma or semicolon separation is accepted.

The order matters: you must click the **same points** in the image in the **same order**.

---

## Tips (things that actually matter)

- **Point order consistency** is more important than “clockwise”.
- Use `--ransac` if one click might be noisy:
  ```bat
  python DXF_2_Imagemasks\dxf2image_create_config.py --ransac
  ```
- If your DXF has multiple layers, restrict using `--layer <name>`.
- Look at the printed reprojection error stats (RMSE / max). If it’s large, your clicks are inconsistent.

---

## Other folders (secondary)

- `Parametric_Mask_Creation/`: Gradio prototypes for parametric “road plane band” masks.
- `Reocuring_Elements/`: Gradio prototype for placing repeated objects (e.g., trees) and exporting a mask.

---

## License

No license is included yet.
