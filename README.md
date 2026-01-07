# Mask_creation_for_generativ_AI

This document combines:
- Installation (Windows)
- DXF_2_Imagemasks workflow usage
- The dependency list (requirements)

Note: This README/documentation was written with the help of ChatGPT.


## 1) Install


Open a Command Prompt / PowerShell in the repo root.

Install Python packages:
    pip install -r requirements.txt




## 2) DXF workflow overview


You provide:
- a street photo
- a DXF (typically polygons/outlines you want as a mask)
- 6 control points in DXF coordinates (CSV/TXT)
- you click the same 6 points in the image (same order)

The scripts compute a homography (optionally RANSAC), project DXF geometry into the image,
and export:
- mask.png (black/white)
- overlay.png (mask on top of the photo)
- optionally *_diag.png (reprojection error vectors)
- optionally *_config.json (re-run without clicking)

Hard limitation:
- Assumes the edited surface is planar (road plane). Works for ground markings/areas.
  Will not correctly map vertical objects (facades, poles, etc.).
- For placing pre-drawn templates in image space using a saved transform, see folder:
  Reocuring_Elements/



## 3) Quickstart (DXF → mask)


From the repo root:

### A) Run with file dialogs (creates a config)
    python DXF_2_Imagemasks\\dxf2image_create_config.py

You will be asked to pick:
- image
- DXF
- output mask path
- (optional) output overlay path
- (optional) output config path
- CSV/TXT file with the 6 DXF control points

Then you click 6 points in the image.

### B) Re-run without clicking (use the saved config)
    python DXF_2_Imagemasks\\dxf2image_load_config.py

Choose:
- image
- config file
- DXF file

This outputs the DXF elements in the image coordinate system and exports mask/overlay again.



## 4) Input format: the 6-point CSV


The CSV/TXT must contain exactly 6 rows of:
    x,y

- Header is allowed.
- Comma or semicolon separation is accepted.

Order matters:
- you must click the same points in the image in the same order as in the CSV.



## 5) Tips


- Point order consistency is the #1 failure mode.
- Keep the DXF clean (only elements that should be projected). Export a minimal DXF if needed.
- Check reprojection error output (RMSE / max). Large errors usually mean inconsistent
  control points (DXF and/or image clicks).
  
  
## 6) Other folders
- Parametric_Mask_Creation/: Gradio prototypes for parametric “road plane band” masks. 
- Reocuring_Elements/: Gradio prototype for placing repeated objects (e.g., trees) and exporting a mask based on the created config file.



## 7) requirements.txt


The repo’s dependency file is 'requirements.txt'. Contents:

numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
ezdxf>=1.1
gradio>=4.0
matplotlib>=3.7
tqdm>=4.66
shapely>=2.0
