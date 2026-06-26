Dataset Cleaning & Labeling Tool
Supplementary software for the manuscript:
> **PAPER_TITLE**
> AUTHOR(S), *JOURNAL_NAME*, YEAR. DOI: `10.xxxx/xxxxx` *(to be added on acceptance)*
A single-window desktop GUI (Tkinter) for cleaning and labeling image datasets. It
combines three steps into one workflow: sharpness/aberration analysis,
SIFT-based overlap (duplicate) detection, and manual labeling, then exports a
consolidated report. Overlap detection is treated as a first pass: because it produces
false positives, every flagged pair is confirmed by a human in the re-check panel.
---
Overview of method
Sharpness / aberration analysis — each image is split into a 3x3 grid; per-patch
Laplacian variance is measured, and the sharpness falloff across patches is computed
as `(patch - max_patch) / max_patch` (most-negative value = worst falloff).
Overlap detection — pairwise SIFT feature matching with Lowe's ratio test, RANSAC
homography, then IoU and containment between warped image masks; a blended overlay is
saved for each flagged pair for visual verification.
Caution flags
Overlap caution when a pair's IoU is >= 5%.
Sharpness caution when the sharpness falloff is > 50%.
Human re-check — each flagged overlap pair is confirmed as Real overlap or
False positive; confirmed duplicates produce removal suggestions (keep the sharper
image).
---
Requirements
Python 3.8+
Packages (`requirements.txt`): `pillow`, `pandas`, `opencv-contrib-python`, `numpy`,
`openpyxl`
> SIFT requires `opencv-contrib-python` (not plain `opencv-python`). Tkinter ships with
> standard Python on Windows/macOS; on some Linux distros install it via
> `sudo apt install python3-tk`.
Exact versions used for the published results can be pinned for reproducibility, e.g.:
```bash
pip freeze > requirements-lock.txt
```
---
Installation
```bash
git clone https://github.com/<your-username>/dataset-cleaning-tool.git
cd dataset-cleaning-tool
python -m venv venv          # optional but recommended
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Usage
```bash
python dataset_cleaning_tool.py
```
Stage 1 — Process. Browse to a main folder (each image subfolder is processed, or the
folder itself if there are no subfolders) and click Process Folder. Outputs per folder:
`<folder>_dataset_report.xlsx` (sheets `Sharpness_Metrics`, `Overlap_Pairs`) and a
`<folder>_overlays/` folder of overlay images. Overlap detection is pairwise (O(n^2)).
Stage 2 — Load & label. Select the generated report and the matching image folder,
then Load Images & Start Labeling. Images load sorted blurriest-first. Label each image,
resolve caution flags, and re-check overlap pairs.
Stage 3 — Finish & report. Finish Session & Generate Report writes a workbook with
`Labels`, `Overlap_Review`, `Removal_Suggestions`, and `Summary` sheets.
Keyboard shortcuts
Key	Action
`q` / `w`	Sharpness: Blur / Sharp
`a` `s` `d` `f`	Anomaly: Poor prep / Damaged / Abnormal / None
`z` / `x`	Identifier: Correct / Mis-labeled
`<-` / `->` or `Enter`	Previous / Next image
`Ctrl+F`	Finish session & save report
Mouse wheel / drag	Zoom / pan
---
Configuration
Thresholds and SIFT parameters are constants at the top of `dataset_cleaning_tool.py`:
```python
IOU_CAUTION_THRESHOLD = 0.05          # >= 5% IoU -> overlap caution
ABERRATION_FALLOFF_THRESHOLD = 0.50   # > 50% falloff -> sharpness caution
```
---
Repository contents
File	Purpose
`dataset_cleaning_tool.py`	The application
`requirements.txt`	Python dependencies
`README.md`	This file
`LICENSE`	MIT license
`CITATION.cff`	Machine-readable citation metadata
`sample_data/`	(optional) small demo dataset for reviewers
---
How to cite
If you use this software, please cite both the article above and the software (see
`CITATION.cff`). For a permanent, version-pinned archive, consider depositing a release on
Zenodo to obtain a DOI (Zenodo integrates directly with GitHub
releases). Many journals prefer a DOI-backed archive over a bare repository link.
License
MIT License — see `LICENSE`.
