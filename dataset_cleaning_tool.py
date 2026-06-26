"""
Dataset Cleaning & Labeling Tool
================================================================================
Merges three workflows into one GUI:

  STAGE 1  - Process a folder:
             * Sharpness / aberration analysis (3x3 Laplacian grid -> falloff)
             * SIFT overlap detection between every image pair (IoU + containment)
             * Saves ONE report workbook per (sub)folder + overlay images

  STAGE 2  - Load report + image folder and label images:
             * Zoom / pan canvas
             * Sharpness (radio)  |  Anomaly (multi-select)  |  Identifier (radio)
             * CAUTION flags:
                 - Overlap caution    : raised when IoU >= 5%
                 - Sharpness caution  : raised when sharpness falloff > 50%
             * Overlap RE-CHECK panel (overlap results contain false positives,
               so every flagged pair must be confirmed: Real overlap / False
               positive). Includes an overlay viewer to inspect each pair.

  STAGE 3  - Finish & generate a consolidated report:
             * Labels, overlap-review decisions, duplicate-removal suggestions,
               and a summary sheet.

--------------------------------------------------------------------------------
Requirements:
    pip install pillow pandas opencv-contrib-python numpy openpyxl
--------------------------------------------------------------------------------
"""

import os
import queue
import threading
from itertools import combinations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkFont

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------- #
#  Tunable thresholds (the two cautions you asked for live here)
# ----------------------------------------------------------------------------- #
IOU_CAUTION_THRESHOLD = 0.05          # >= 5% IoU -> overlap caution
ABERRATION_FALLOFF_THRESHOLD = 0.50   # > 50% sharpness falloff -> sharpness caution

# SIFT / overlap parameters
SIFT_NFEATURES = 5000
RATIO_TEST = 0.75
MIN_MATCHES = 20
MIN_INLIERS = 15
# Auto false-positive heuristic (high containment + low inliers => suspicious)
DEF_CONTAINMENT_FP = 1.00
POSS_CONTAINMENT_FP = 0.90
FP_INLIER_THRESHOLD = 400

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

REPORT_SUFFIX = "_dataset_report.xlsx"
OVERLAY_SUFFIX = "_overlays"


# ============================================================================= #
#  Sharpness / aberration helpers
# ============================================================================= #
def analyze_local_sharpness(image, grid_size=(3, 3)):
    """Return (flat list of 9 patch Laplacian variances, max patch variance)."""
    if image is None:
        raise ValueError("Image is None.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    ph, pw = height // grid_size[0], width // grid_size[1]
    values = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y0, y1 = i * ph, (i + 1) * ph
            x0, x1 = j * pw, (j + 1) * pw
            if i == grid_size[0] - 1:
                y1 = height
            if j == grid_size[1] - 1:
                x1 = width
            patch = gray[y0:y1, x0:x1]
            values[i, j] = 0.0 if patch.size == 0 else cv2.Laplacian(patch, cv2.CV_64F).var()
    flat = values.flatten().tolist()
    return flat, (float(np.max(values)) if values.size else 0.0)


def min_aberration_from_patches(patch_values):
    """
    Aberration = (patch - max_patch) / max_patch  (always <= 0).
    The most negative value (min) is the worst sharpness falloff across the image.
    Returns NaN when the image has no sharpness signal.
    """
    arr = np.asarray(patch_values, dtype=float)
    mx = np.max(arr) if arr.size else 0.0
    if mx == 0:
        return float('nan')
    aberr = (arr - mx) / mx
    return float(np.min(aberr))


# ============================================================================= #
#  SIFT overlap helpers
# ============================================================================= #
def compute_features(image, detector):
    if image is None:
        return None, None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        return (kp if kp is not None else []), des
    except Exception:
        return None, None


def check_stitchable(good_matches, kp1, kp2, min_inliers=MIN_INLIERS):
    if not kp1 or not kp2 or len(good_matches) < max(4, min_inliers):
        return False, 0, None
    try:
        src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return False, 0, None
        inliers = int(mask.sum())
        return inliers >= min_inliers, inliers, H
    except Exception:
        return False, 0, None


def create_full_overlay(img1, img2, H):
    """Warp img1 into img2's frame; return (overlay_bgr, iou, containment)."""
    if img1 is None or img2 is None or H is None:
        return None, 0.0, 0.0
    try:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        c1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], np.float32).reshape(-1, 1, 2)
        warp_c1 = cv2.perspectiveTransform(c1, H)
        c2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], np.float32).reshape(-1, 1, 2)
        all_c = np.vstack((warp_c1, c2))
        x_min, y_min = np.int32(all_c.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_c.max(axis=0).ravel() + 0.5)
        cw, ch = x_max - x_min, y_max - y_min
        if cw <= 0 or ch <= 0 or cw * ch > 50_000_000:   # guard against runaway canvas
            return None, 0.0, 0.0
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], np.float32)
        warp1 = cv2.warpPerspective(img1, T @ H, (cw, ch))
        warp2 = cv2.warpPerspective(img2, T, (cw, ch))
        m1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY) > 0
        m2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY) > 0
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        iou = float(inter) / union if union > 0 else 0.0
        area1 = m1.sum()
        containment = float(inter) / area1 if area1 > 0 else 0.0
        overlay = cv2.addWeighted(warp2, 0.5, warp1, 0.5, 0)
        return overlay, iou, containment
    except Exception:
        return None, 0.0, 0.0


# ============================================================================= #
#  Main application
# ============================================================================= #
class DatasetCleaningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Cleaning & Labeling Tool")
        self.root.geometry("1280x960")
        self.root.minsize(1100, 820)

        # --- shared state ---
        self.main_folder_var = tk.StringVar()
        self.report_excel_var = tk.StringVar()
        self.image_folder_var = tk.StringVar()
        self.output_log_name_var = tk.StringVar(value="dataset_clean_log.xlsx")

        self.sorted_image_info_list = []
        self.current_image_index = 0

        # overlap data:  name -> list of partner dicts ;  ('a','b') -> review decision
        self.overlap_by_image = {}
        self.overlap_reviews = {}
        self.overlay_folder_hint = ""

        # image / zoom state
        self.current_pil_image = None
        self.photo_img = None
        self.zoom_factor = 1.0

        # label vars
        self.sharpness_label_var = tk.StringVar(value="")
        self.identifier_label_var = tk.StringVar(value="")
        self.anomaly_vars = {
            "POOR PREPARATION": tk.BooleanVar(value=False),
            "DAMAGED SPECIMEN": tk.BooleanVar(value=False),
            "ABNORMAL ANATOMY": tk.BooleanVar(value=False),
            "NONE": tk.BooleanVar(value=False),
        }
        self.anomaly_key_map = {
            "a": "POOR PREPARATION", "s": "DAMAGED SPECIMEN",
            "d": "ABNORMAL ANATOMY", "f": "NONE",
        }

        self.info_font = tkFont.Font(family="Helvetica", size=10, weight="bold")
        self.caution_font = tkFont.Font(family="Helvetica", size=11, weight="bold")
        self.max_display_width = 800
        self.max_display_height = 600

        self._msg_queue = queue.Queue()

        self._build_setup_bar()
        self._build_main_area()
        self._build_bottom_bar()

        self._set_labeling_active(False)
        self._setup_shortcuts()

    # ------------------------------------------------------------------ UI build
    def _build_setup_bar(self):
        f = ttk.Frame(self.root, padding="10")
        f.pack(side=tk.TOP, fill=tk.X)

        # Stage 1
        ttk.Label(f, text="STAGE 1 - Process folder", font=self.info_font).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Label(f, text="Main folder:").grid(row=1, column=0, padx=5, pady=4, sticky=tk.W)
        self.main_folder_entry = ttk.Entry(f, textvariable=self.main_folder_var, width=60)
        self.main_folder_entry.grid(row=1, column=1, padx=5, pady=4, sticky=tk.EW)
        self.browse_main_btn = ttk.Button(f, text="Browse...", command=self.select_main_folder)
        self.browse_main_btn.grid(row=1, column=2, padx=5, pady=4)
        self.process_btn = ttk.Button(
            f, text="Process Folder (Sharpness + Overlap Detection)",
            command=self.run_processing)
        self.process_btn.grid(row=2, column=0, columnspan=3, pady=(4, 6))

        self.progress = ttk.Progressbar(f, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=(0, 4))

        ttk.Separator(f, orient='horizontal').grid(
            row=4, column=0, columnspan=3, sticky='ew', pady=6)

        # Stage 2
        ttk.Label(f, text="STAGE 2 - Load & label", font=self.info_font).grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Label(f, text="Report Excel:").grid(row=6, column=0, padx=5, pady=4, sticky=tk.W)
        self.report_entry = ttk.Entry(f, textvariable=self.report_excel_var, width=60)
        self.report_entry.grid(row=6, column=1, padx=5, pady=4, sticky=tk.EW)
        self.browse_report_btn = ttk.Button(f, text="Browse...", command=self.select_report_excel)
        self.browse_report_btn.grid(row=6, column=2, padx=5, pady=4)
        ttk.Label(f, text="Image folder:").grid(row=7, column=0, padx=5, pady=4, sticky=tk.W)
        self.image_folder_entry = ttk.Entry(f, textvariable=self.image_folder_var, width=60)
        self.image_folder_entry.grid(row=7, column=1, padx=5, pady=4, sticky=tk.EW)
        self.browse_img_btn = ttk.Button(f, text="Browse...", command=self.select_image_folder)
        self.browse_img_btn.grid(row=7, column=2, padx=5, pady=4)
        self.start_btn = ttk.Button(f, text="Load Images & Start Labeling",
                                    command=self.load_and_sort)
        self.start_btn.grid(row=8, column=0, columnspan=3, pady=(6, 2))

        f.columnconfigure(1, weight=1)
        self.setup_frame = f

    def _build_main_area(self):
        main = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- left: image + metrics ----
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        disp = ttk.Frame(left)
        disp.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(disp, bg="#2b2b2b", cursor="fleur", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        zoom = ttk.Frame(disp)
        zoom.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.zoom_out_btn = ttk.Button(zoom, text="Zoom -", state=tk.DISABLED,
                                       command=lambda: self.adjust_zoom(0.8))
        self.zoom_out_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.zoom_reset_btn = ttk.Button(zoom, text="Reset Zoom", state=tk.DISABLED,
                                         command=lambda: self.adjust_zoom(0))
        self.zoom_reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.zoom_in_btn = ttk.Button(zoom, text="Zoom +", state=tk.DISABLED,
                                      command=lambda: self.adjust_zoom(1.2))
        self.zoom_in_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # caution banner
        self.caution_label = ttk.Label(left, text="", anchor=tk.W, justify=tk.LEFT,
                                       font=self.caution_font, foreground="#b00020")
        self.caution_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        self.image_info_label = ttk.Label(
            left, text="Status: Process a folder (Stage 1) or load a report (Stage 2).",
            padding=(5, 5), anchor=tk.W, justify=tk.LEFT, font=self.info_font)
        self.image_info_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        # ---- right: labeling + overlap review ----
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ctrl = ttk.Frame(right)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        self.sharpness_frame = ttk.LabelFrame(ctrl, text="Sharpness  (q: Blur,  w: Sharp)")
        self.sharpness_frame.pack(fill=tk.X, padx=5, pady=(8, 4))
        for opt in ["BLUR", "SHARP"]:
            ttk.Radiobutton(self.sharpness_frame, text=opt, value=opt,
                            variable=self.sharpness_label_var,
                            command=lambda o=opt: self.record_label('sharpness', o)
                            ).pack(side=tk.LEFT, padx=10, pady=6, expand=True, fill=tk.X)

        self.anomalous_frame = ttk.LabelFrame(
            ctrl, text="Anomaly - multi-select  (a, s, d, f)")
        self.anomalous_frame.pack(fill=tk.X, padx=5, pady=4)
        for opt in self.anomaly_vars:
            ttk.Checkbutton(self.anomalous_frame, text=opt, variable=self.anomaly_vars[opt],
                            command=lambda o=opt: self.record_anomaly(o)
                            ).pack(side=tk.LEFT, padx=4, pady=6, expand=True)

        self.identifier_frame = ttk.LabelFrame(
            ctrl, text="Identifier  (z: Correct,  x: Mis-labeled)")
        self.identifier_frame.pack(fill=tk.X, padx=5, pady=4)
        for opt in ["CORRECT LABEL", "MIS-LABELED"]:
            ttk.Radiobutton(self.identifier_frame, text=opt, value=opt,
                            variable=self.identifier_label_var,
                            command=lambda o=opt: self.record_label('identifier_accuracy', o)
                            ).pack(side=tk.LEFT, padx=10, pady=6, expand=True, fill=tk.X)

        # overlap re-check (scrollable)
        rev_outer = ttk.LabelFrame(right, text="Overlap Re-check  (confirm real overlaps vs false positives)")
        rev_outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 4))
        rev_canvas = tk.Canvas(rev_outer, highlightthickness=0)
        rev_scroll = ttk.Scrollbar(rev_outer, orient="vertical", command=rev_canvas.yview)
        self.overlap_review_frame = ttk.Frame(rev_canvas)
        self.overlap_review_frame.bind(
            "<Configure>", lambda e: rev_canvas.configure(scrollregion=rev_canvas.bbox("all")))
        rev_canvas.create_window((0, 0), window=self.overlap_review_frame, anchor="nw")
        rev_canvas.configure(yscrollcommand=rev_scroll.set)
        rev_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rev_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # navigation
        nav = ttk.Frame(right)
        nav.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 4), padx=5)
        self.previous_button = ttk.Button(nav, text="<< Previous (Left)",
                                          command=self.go_to_previous, state=tk.DISABLED)
        self.previous_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=8)
        self.next_button = ttk.Button(nav, text="Next (Right / Enter) >>",
                                      command=self.go_to_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5, ipady=8)

    def _build_bottom_bar(self):
        f = ttk.Frame(self.root, padding=(10, 5, 10, 10))
        f.pack(side=tk.BOTTOM, fill=tk.X)
        self.finished_button = ttk.Button(
            f, text="Finish Session & Generate Report  (Ctrl+F)",
            command=self.on_finish_click, state=tk.DISABLED)
        self.finished_button.pack(expand=True, fill=tk.X, padx=5, pady=5, ipady=10)

    # ------------------------------------------------------- keyboard shortcuts
    def _setup_shortcuts(self):
        self.root.bind('q', lambda e: self._key_label('sharpness', "BLUR", self.sharpness_label_var))
        self.root.bind('w', lambda e: self._key_label('sharpness', "SHARP", self.sharpness_label_var))
        for k in ('a', 's', 'd', 'f'):
            self.root.bind(k, lambda e, key=k: self._key_anomaly(key))
        self.root.bind('z', lambda e: self._key_label('identifier_accuracy', "CORRECT LABEL", self.identifier_label_var))
        self.root.bind('x', lambda e: self._key_label('identifier_accuracy', "MIS-LABELED", self.identifier_label_var))
        self.root.bind('<Left>', lambda e: self.previous_button.instate(['!disabled']) and self.go_to_previous())
        self.root.bind('<Right>', lambda e: self.next_button.instate(['!disabled']) and self.go_to_next())
        self.root.bind('<Return>', lambda e: self.next_button.instate(['!disabled']) and self.go_to_next())
        self.root.bind('<Control-f>', lambda e: self.finished_button.instate(['!disabled']) and self.on_finish_click())
        self.root.bind('<Control-F>', lambda e: self.finished_button.instate(['!disabled']) and self.on_finish_click())

    def _labeling_enabled(self):
        return bool(self.sorted_image_info_list) and self.finished_button.instate(['!disabled'])

    def _key_label(self, cat, value, var):
        if self._labeling_enabled():
            var.set(value)
            self.record_label(cat, value)

    def _key_anomaly(self, key):
        if self._labeling_enabled():
            opt = self.anomaly_key_map[key]
            self.anomaly_vars[opt].set(not self.anomaly_vars[opt].get())
            self.record_anomaly(opt)

    # ============================================================ STAGE 1
    def select_main_folder(self):
        path = filedialog.askdirectory(title="Select main folder (with image subfolders)")
        if path:
            self.main_folder_var.set(path)
            self._status("Main folder selected. Click 'Process Folder'.")

    def run_processing(self):
        main_folder = self.main_folder_var.get()
        if not main_folder or not os.path.isdir(main_folder):
            messagebox.showerror("Error", "Select a valid main folder first.")
            return
        if not messagebox.askyesno(
                "Confirm",
                "Process all images (sharpness + pairwise SIFT overlap)?\n\n"
                "Overlap detection is pairwise (O(n^2)) and may take a while for "
                "large folders. The window stays responsive; a progress bar shows status."):
            return

        for w in (self.process_btn, self.browse_main_btn, self.main_folder_entry):
            w.config(state=tk.DISABLED)
        self.progress.config(value=0)

        t = threading.Thread(target=self._process_worker, args=(main_folder,), daemon=True)
        t.start()
        self.root.after(100, self._poll_messages)

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == 'status':
                    self.image_info_label.config(text=f"Status: {payload}")
                elif kind == 'progress':
                    done, total = payload
                    self.progress.config(maximum=max(total, 1), value=done)
                elif kind == 'done':
                    self._on_processing_done(payload)
                    return
                elif kind == 'error':
                    messagebox.showerror("Processing Error", payload)
                    for w in (self.process_btn, self.browse_main_btn, self.main_folder_entry):
                        w.config(state=tk.NORMAL)
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _q(self, kind, payload):
        self._msg_queue.put((kind, payload))

    def _process_worker(self, main_folder):
        try:
            subdirs = [os.path.join(main_folder, d) for d in os.listdir(main_folder)
                       if os.path.isdir(os.path.join(main_folder, d))
                       and not d.endswith(OVERLAY_SUFFIX)]
            targets = subdirs if subdirs else [main_folder]
            any_done = False
            for idx, folder in enumerate(targets):
                self._q('status', f"[{idx+1}/{len(targets)}] Processing {os.path.basename(folder)} ...")
                if self._process_one_folder(folder):
                    any_done = True
            self._q('done', any_done)
        except Exception as e:
            self._q('error', str(e))

    def _process_one_folder(self, folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)]
        if not images:
            return False

        # ---- sharpness / aberration ----
        sharpness_rows = []
        feats = {}            # path -> (kp, des, shape)
        sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

        total_steps = len(images)
        for i, name in enumerate(images):
            path = os.path.join(folder, name)
            img = cv2.imread(path)
            if img is None:
                continue
            patches, max_lap = analyze_local_sharpness(img)
            min_aberr = min_aberration_from_patches(patches)
            falloff = abs(min_aberr) if not np.isnan(min_aberr) else np.nan
            sharpness_rows.append({
                'Image Name': name,
                **{f'Patch_{k}': v for k, v in enumerate(patches)},
                'Max Patch Laplacian Var': max_lap,
                'Min Aberration': min_aberr,
                'Sharpness Falloff %': falloff * 100 if not np.isnan(falloff) else np.nan,
                'Sharpness Caution': 'YES' if (not np.isnan(falloff) and falloff > ABERRATION_FALLOFF_THRESHOLD) else 'no',
            })
            kp, des = compute_features(img, sift)
            if des is not None and len(des) >= 2:
                feats[path] = (kp, des, img.shape[:2])
            self._q('progress', (i + 1, total_steps))
            self._q('status', f"Sharpness: {os.path.basename(folder)} {i+1}/{total_steps}")

        # ---- SIFT overlap ----
        folder_name = os.path.basename(os.path.normpath(folder))
        overlay_folder = os.path.join(folder, f"{folder_name}{OVERLAY_SUFFIX}")
        os.makedirs(overlay_folder, exist_ok=True)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        valid_paths = list(feats.keys())
        pairs = list(combinations(valid_paths, 2))
        overlap_rows = []
        for p, (path1, path2) in enumerate(pairs):
            self._q('progress', (p + 1, len(pairs)))
            self._q('status', f"Overlap: {folder_name} pair {p+1}/{len(pairs)}")
            kp1, des1, _ = feats[path1]
            kp2, des2, _ = feats[path2]
            b1, b2 = os.path.basename(path1), os.path.basename(path2)
            try:
                matches = flann.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
            except Exception:
                continue
            if len(good) < MIN_MATCHES:
                continue
            ok, inliers, H = check_stitchable(good, kp1, kp2, MIN_INLIERS)
            if not ok or H is None:
                continue

            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            overlay, iou, containment = create_full_overlay(img1, img2, H)
            if iou < IOU_CAUTION_THRESHOLD:
                continue  # only record pairs that reach the caution threshold

            # auto false-positive hint
            if containment >= DEF_CONTAINMENT_FP and inliers < FP_INLIER_THRESHOLD:
                auto_fp = "DEFINITE FP?"
            elif containment >= POSS_CONTAINMENT_FP and inliers < FP_INLIER_THRESHOLD:
                auto_fp = "POSSIBLE FP?"
            else:
                auto_fp = "no"

            overlay_path = ""
            if overlay is not None:
                fname = f"{os.path.splitext(b1)[0]}__{os.path.splitext(b2)[0]}_iou{int(iou*100)}.jpg"
                overlay_path = os.path.join(overlay_folder, fname)
                try:
                    cv2.imwrite(overlay_path, overlay)
                except Exception:
                    overlay_path = ""

            overlap_rows.append({
                'Image A': b1,
                'Image B': b2,
                'IoU %': round(iou * 100, 2),
                'Containment %': round(containment * 100, 2),
                'SIFT Matches': len(good),
                'Inliers': inliers,
                'Auto FP Flag': auto_fp,
                'Overlay File': overlay_path,
            })

        # ---- save report workbook ----
        report_path = os.path.join(folder, f"{folder_name}{REPORT_SUFFIX}")
        with pd.ExcelWriter(report_path, engine='openpyxl') as xl:
            pd.DataFrame(sharpness_rows).to_excel(xl, sheet_name="Sharpness_Metrics", index=False)
            ov_df = pd.DataFrame(overlap_rows) if overlap_rows else pd.DataFrame(
                columns=['Image A', 'Image B', 'IoU %', 'Containment %', 'SIFT Matches',
                         'Inliers', 'Auto FP Flag', 'Overlay File'])
            ov_df.to_excel(xl, sheet_name="Overlap_Pairs", index=False)
        self._q('status', f"Saved report: {os.path.basename(report_path)}")
        return True

    def _on_processing_done(self, any_done):
        if any_done:
            self.process_btn.config(state=tk.DISABLED)
            messagebox.showinfo(
                "Done",
                "Sharpness + overlap processing complete.\n\n"
                "Each folder now has a *_dataset_report.xlsx and an _overlays folder.\n"
                "Select a report Excel and its image folder (Stage 2) to start labeling.")
            self._status("Processing complete. Load a report + image folder (Stage 2).")
        else:
            for w in (self.process_btn, self.browse_main_btn, self.main_folder_entry):
                w.config(state=tk.NORMAL)
            messagebox.showwarning("Nothing processed", "No valid images were found.")
            self._status("No images processed.")

    # ============================================================ STAGE 2
    def select_report_excel(self):
        init = self.main_folder_var.get() if os.path.isdir(self.main_folder_var.get() or "") else "."
        path = filedialog.askopenfilename(
            title="Select *_dataset_report.xlsx", initialdir=init,
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
        if path:
            self.report_excel_var.set(path)
            self._status("Report Excel selected.")

    def select_image_folder(self):
        init = self.main_folder_var.get() if os.path.isdir(self.main_folder_var.get() or "") else "."
        path = filedialog.askdirectory(title="Select image folder for labeling", initialdir=init)
        if path:
            self.image_folder_var.set(path)
            self._status("Image folder selected.")

    def load_and_sort(self):
        excel_path = self.report_excel_var.get()
        img_folder = self.image_folder_var.get()
        if not excel_path or not img_folder:
            messagebox.showerror("Error", "Select both the report Excel and the image folder.")
            return
        if not os.path.exists(excel_path):
            messagebox.showerror("Error", f"Report not found: {excel_path}")
            return
        if not os.path.isdir(img_folder):
            messagebox.showerror("Error", f"Image folder not found: {img_folder}")
            return

        # read sharpness metrics
        try:
            df = pd.read_excel(excel_path, sheet_name="Sharpness_Metrics")
        except Exception:
            try:
                df = pd.read_excel(excel_path)  # fall back to first sheet
            except Exception as e:
                messagebox.showerror("Error", f"Could not read report: {e}")
                return
        if 'Image Name' not in df.columns:
            messagebox.showerror("Error", "Report must contain an 'Image Name' column.")
            return

        # read overlap pairs (optional)
        self.overlap_by_image, self.overlap_reviews = {}, {}
        try:
            ov = pd.read_excel(excel_path, sheet_name="Overlap_Pairs")
        except Exception:
            ov = pd.DataFrame()
        for _, r in ov.iterrows():
            a, b = str(r['Image A']), str(r['Image B'])
            rec_a = {'partner': b, 'iou': float(r.get('IoU %', 0)),
                     'containment': float(r.get('Containment %', 0)),
                     'inliers': int(r.get('Inliers', 0)),
                     'auto_fp': str(r.get('Auto FP Flag', 'no')),
                     'overlay': str(r.get('Overlay File', '') or '')}
            rec_b = dict(rec_a, partner=a)
            self.overlap_by_image.setdefault(a, []).append(rec_a)
            self.overlap_by_image.setdefault(b, []).append(rec_b)
            self.overlap_reviews[self._pair_key(a, b)] = "Unreviewed"

        # choose sort key: blurriest first
        sort_col = None
        for cand in ('Max Patch Laplacian Var',):
            if cand in df.columns:
                df[cand] = pd.to_numeric(df[cand], errors='coerce')
                if not df[cand].isnull().all():
                    sort_col = cand
                    break
        df_sorted = df.sort_values(by=sort_col, ascending=True, na_position='last') if sort_col else df

        self.sorted_image_info_list = []
        for _, row in df_sorted.iterrows():
            name = str(row['Image Name'])
            path = os.path.join(img_folder, name)
            if not os.path.exists(path):
                print(f"Warning: image not found, skipping: {name}")
                continue
            falloff = row.get('Sharpness Falloff %', np.nan)
            self.sorted_image_info_list.append({
                'name': name, 'path': path,
                'max_lap': row.get('Max Patch Laplacian Var', np.nan),
                'min_aberr': row.get('Min Aberration', np.nan),
                'falloff_pct': falloff,
                'labels': {'sharpness': None, 'anomaly': [], 'identifier_accuracy': None},
            })

        if not self.sorted_image_info_list:
            messagebox.showinfo("No images", "No valid images found from the report.")
            return

        self.current_image_index = 0
        for w in (self.start_btn, self.report_entry, self.browse_report_btn,
                  self.image_folder_entry, self.browse_img_btn):
            w.config(state=tk.DISABLED)
        self._set_labeling_active(True)
        self.display_current_image()

    # ---------------------------------------------------------- image display
    def display_current_image(self):
        if not self.sorted_image_info_list:
            return
        info = self.sorted_image_info_list[self.current_image_index]
        try:
            self.current_pil_image = Image.open(info['path'])
            self.zoom_factor = 1.0
            self.render_image()
            for w in (self.zoom_out_btn, self.zoom_reset_btn, self.zoom_in_btn):
                w.config(state=tk.NORMAL)

            labels = info['labels']
            self.sharpness_label_var.set(labels.get('sharpness') or "")
            self.identifier_label_var.set(labels.get('identifier_accuracy') or "")
            saved_anoms = labels.get('anomaly') or []
            for opt, var in self.anomaly_vars.items():
                var.set(opt in saved_anoms)

            self._update_cautions(info)
            self._build_overlap_review(info)
            self._update_navigation()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not display {info['name']}: {e}")

    def render_image(self):
        if not self.current_pil_image:
            return
        self.canvas.update_idletasks()
        fw, fh = self.canvas.winfo_width(), self.canvas.winfo_height()
        tw = min(fw if fw > 10 else self.max_display_width, self.max_display_width)
        th = min(fh if fh > 10 else self.max_display_height, self.max_display_height)
        iw, ih = self.current_pil_image.size
        ratio = 1.0
        if iw > tw:
            ratio = tw / iw
        if ih * ratio > th:
            ratio = th / ih
        nw = max(1, int(iw * ratio * self.zoom_factor))
        nh = max(1, int(ih * ratio * self.zoom_factor))
        resized = self.current_pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photo_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(max(fw // 2, nw // 2), max(fh // 2, nh // 2),
                                 anchor=tk.CENTER, image=self.photo_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self._update_info_label(self.sorted_image_info_list[self.current_image_index])

    def on_mouse_wheel(self, event):
        if not self.current_pil_image:
            return
        if getattr(event, 'delta', 0):
            self.adjust_zoom(1.1 if event.delta > 0 else 0.9)
        elif getattr(event, 'num', None) == 4:
            self.adjust_zoom(1.1)
        elif getattr(event, 'num', None) == 5:
            self.adjust_zoom(0.9)

    def adjust_zoom(self, mult):
        if not self.current_pil_image:
            return
        self.zoom_factor = 1.0 if mult == 0 else self.zoom_factor * mult
        self.zoom_factor = max(0.1, min(self.zoom_factor, 12.0))
        self.render_image()

    # ---------------------------------------------------------- cautions
    def _update_cautions(self, info):
        msgs = []
        falloff = info.get('falloff_pct', np.nan)
        if isinstance(falloff, (int, float)) and not np.isnan(falloff) and falloff > ABERRATION_FALLOFF_THRESHOLD * 100:
            msgs.append(f"\u26A0 SHARPNESS CAUTION: falloff {falloff:.0f}% (> {int(ABERRATION_FALLOFF_THRESHOLD*100)}%)")
        partners = self.overlap_by_image.get(info['name'], [])
        flagged = [p for p in partners if p['iou'] >= IOU_CAUTION_THRESHOLD * 100]
        if flagged:
            max_iou = max(p['iou'] for p in flagged)
            msgs.append(f"\u26A0 OVERLAP CAUTION: {len(flagged)} pair(s), max IoU {max_iou:.0f}% "
                        f"(>= {int(IOU_CAUTION_THRESHOLD*100)}%) - re-check below")
        self.caution_label.config(text="    ".join(msgs))

    def _update_info_label(self, info):
        n = len(self.sorted_image_info_list)
        max_lap = info.get('max_lap', np.nan)
        max_lap_txt = f"{max_lap:.1f}" if isinstance(max_lap, (int, float)) and not np.isnan(max_lap) else "N/A"
        fall = info.get('falloff_pct', np.nan)
        fall_txt = f"{fall:.0f}%" if isinstance(fall, (int, float)) and not np.isnan(fall) else "N/A"
        self.image_info_label.config(
            text=f"File: {info['name']}  ({self.current_image_index+1}/{n})  |  "
                 f"MaxLapVar: {max_lap_txt}  |  Falloff: {fall_txt}  |  Zoom: {int(self.zoom_factor*100)}%")

    # ---------------------------------------------------------- overlap review
    def _pair_key(self, a, b):
        return (a, b) if a <= b else (b, a)

    def _build_overlap_review(self, info):
        for w in self.overlap_review_frame.winfo_children():
            w.destroy()
        partners = [p for p in self.overlap_by_image.get(info['name'], [])
                    if p['iou'] >= IOU_CAUTION_THRESHOLD * 100]
        if not partners:
            ttk.Label(self.overlap_review_frame,
                      text="No overlap pairs at/above the IoU threshold for this image.",
                      foreground="#2e7d32").pack(anchor=tk.W, padx=6, pady=6)
            return

        for p in sorted(partners, key=lambda x: -x['iou']):
            key = self._pair_key(info['name'], p['partner'])
            row = ttk.Frame(self.overlap_review_frame, relief=tk.GROOVE, borderwidth=1)
            row.pack(fill=tk.X, padx=4, pady=4)
            hdr = (f"\u2194 {p['partner']}\n"
                   f"IoU {p['iou']:.0f}%  |  Containment {p['containment']:.0f}%  |  "
                   f"Inliers {p['inliers']}  |  Auto: {p['auto_fp']}")
            ttk.Label(row, text=hdr, justify=tk.LEFT).pack(anchor=tk.W, padx=6, pady=(4, 2))

            decision_var = tk.StringVar(value=self.overlap_reviews.get(key, "Unreviewed"))
            btnrow = ttk.Frame(row)
            btnrow.pack(fill=tk.X, padx=6, pady=(0, 4))
            for label in ("Real overlap", "False positive"):
                ttk.Radiobutton(btnrow, text=label, value=label, variable=decision_var,
                                command=lambda k=key, v=decision_var: self._record_review(k, v)
                                ).pack(side=tk.LEFT, padx=4)
            ttk.Button(btnrow, text="View overlay",
                       command=lambda ov=p['overlay']: self._view_overlay(ov)
                       ).pack(side=tk.RIGHT, padx=4)

    def _record_review(self, key, var):
        self.overlap_reviews[key] = var.get()

    def _view_overlay(self, overlay_path):
        if not overlay_path or not os.path.exists(overlay_path):
            messagebox.showinfo("Overlay", "Overlay image not found on disk.")
            return
        try:
            top = tk.Toplevel(self.root)
            top.title(os.path.basename(overlay_path))
            img = Image.open(overlay_path)
            img.thumbnail((1000, 800), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(top, image=photo)
            lbl.image = photo
            lbl.pack()
        except Exception as e:
            messagebox.showerror("Overlay", f"Could not open overlay: {e}")

    # ---------------------------------------------------------- labeling state
    def record_label(self, cat, value):
        if self.sorted_image_info_list:
            self.sorted_image_info_list[self.current_image_index]['labels'][cat] = value
            self._update_navigation()

    def record_anomaly(self, changed):
        if not self.sorted_image_info_list:
            return
        if changed == "NONE" and self.anomaly_vars["NONE"].get():
            for opt in self.anomaly_vars:
                if opt != "NONE":
                    self.anomaly_vars[opt].set(False)
        elif changed != "NONE" and self.anomaly_vars[changed].get():
            self.anomaly_vars["NONE"].set(False)
        selected = [o for o, v in self.anomaly_vars.items() if v.get()]
        self.sorted_image_info_list[self.current_image_index]['labels']['anomaly'] = selected
        self._update_navigation()

    def _labels_complete(self, info=None):
        info = info or self.sorted_image_info_list[self.current_image_index]
        labels = info['labels']
        return (labels.get('sharpness') and labels.get('anomaly')
                and labels.get('identifier_accuracy'))

    def _update_navigation(self):
        if not self._labeling_enabled():
            self.previous_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return
        self.previous_button.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        is_last = self.current_image_index >= len(self.sorted_image_info_list) - 1
        if is_last:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL if self._labels_complete() else tk.DISABLED)

    def go_to_previous(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()

    def go_to_next(self):
        if not self._labels_complete():
            messagebox.showwarning("Incomplete", "Select a label in all three categories first.")
            return
        if self.current_image_index < len(self.sorted_image_info_list) - 1:
            self.current_image_index += 1
            self.display_current_image()

    def _set_labeling_active(self, active):
        state = tk.NORMAL if active else tk.DISABLED
        for frame in (self.sharpness_frame, self.anomalous_frame, self.identifier_frame):
            for w in frame.winfo_children():
                if isinstance(w, (ttk.Radiobutton, ttk.Checkbutton)):
                    w.config(state=state)
        self.finished_button.config(state=state)
        if not active:
            self.previous_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)

    # ============================================================ STAGE 3
    def on_finish_click(self):
        if not self.sorted_image_info_list:
            return
        incomplete = [i for i, inf in enumerate(self.sorted_image_info_list)
                      if not self._labels_complete(inf)]
        unreviewed = [k for k, v in self.overlap_reviews.items() if v == "Unreviewed"]

        warn = []
        if incomplete:
            warn.append(f"{len(incomplete)} image(s) not fully labeled (will be 'Unlabeled').")
        if unreviewed:
            warn.append(f"{len(unreviewed)} overlap pair(s) not re-checked (will be 'Unreviewed').")
        if warn and not messagebox.askyesno(
                "Confirm Finish", "\n".join(warn) + "\n\nProceed and generate the report?"):
            return
        if not warn and not messagebox.askyesno("Confirm Finish",
                                                "Save all labels and generate the report?"):
            return

        self.save_report()
        self._reset_ui()

    def save_report(self):
        log_name = self.output_log_name_var.get() or "dataset_clean_log.xlsx"
        if not log_name.lower().endswith(".xlsx"):
            log_name += ".xlsx"
        out_dir = os.path.dirname(self.report_excel_var.get()) or os.getcwd()
        out_path = os.path.join(out_dir, log_name)

        # sharpness lookup for removal suggestions
        sharp_lookup = {inf['name']: inf.get('max_lap', np.nan) for inf in self.sorted_image_info_list}

        # --- Labels sheet ---
        label_rows = []
        for inf in self.sorted_image_info_list:
            labels = inf['labels']
            anoms = labels.get('anomaly') or []
            partners = [p for p in self.overlap_by_image.get(inf['name'], [])
                        if p['iou'] >= IOU_CAUTION_THRESHOLD * 100]
            max_iou = max((p['iou'] for p in partners), default=0.0)
            fall = inf.get('falloff_pct', np.nan)
            sharp_caut = (isinstance(fall, (int, float)) and not np.isnan(fall)
                          and fall > ABERRATION_FALLOFF_THRESHOLD * 100)
            label_rows.append({
                'Image_Name': inf['name'],
                'Max_Patch_Laplacian_Var': inf.get('max_lap', np.nan),
                'Min_Aberration': inf.get('min_aberr', np.nan),
                'Sharpness_Falloff_%': fall,
                'Sharpness_Caution': 'YES' if sharp_caut else 'no',
                'Overlap_Caution': 'YES' if partners else 'no',
                'Max_IoU_%': round(max_iou, 1),
                'Num_Overlap_Pairs': len(partners),
                'Sharpness': labels.get('sharpness') or 'Unlabeled',
                'Anomaly': ", ".join(anoms) if anoms else 'Unlabeled',
                'Identifier_Accuracy': labels.get('identifier_accuracy') or 'Unlabeled',
            })

        # --- Overlap review sheet + removal suggestions ---
        review_rows, removal_rows = [], []
        for (a, b), decision in self.overlap_reviews.items():
            rec = next((p for p in self.overlap_by_image.get(a, []) if p['partner'] == b), {})
            review_rows.append({
                'Image_A': a, 'Image_B': b,
                'IoU_%': rec.get('iou', np.nan),
                'Containment_%': rec.get('containment', np.nan),
                'Inliers': rec.get('inliers', np.nan),
                'Auto_FP_Flag': rec.get('auto_fp', ''),
                'Review_Decision': decision,
            })
            if decision == "Real overlap":
                la, lb = sharp_lookup.get(a, np.nan), sharp_lookup.get(b, np.nan)
                # suggest removing the blurrier (lower max-lap-var) image
                if (isinstance(la, (int, float)) and isinstance(lb, (int, float))
                        and not np.isnan(la) and not np.isnan(lb)):
                    remove = a if la < lb else b
                    keep = b if remove == a else a
                else:
                    remove, keep = b, a
                removal_rows.append({'Keep': keep, 'Suggested_Remove': remove,
                                     'Reason': f"Real overlap, removed image is blurrier"})

        # --- Summary sheet ---
        sharp_counts = pd.Series([r['Sharpness'] for r in label_rows]).value_counts().to_dict()
        summary_rows = [
            {'Metric': 'Total images', 'Value': len(label_rows)},
            {'Metric': 'Images flagged SHARPNESS caution', 'Value': sum(1 for r in label_rows if r['Sharpness_Caution'] == 'YES')},
            {'Metric': 'Images flagged OVERLAP caution', 'Value': sum(1 for r in label_rows if r['Overlap_Caution'] == 'YES')},
            {'Metric': 'Overlap pairs (IoU >= 5%)', 'Value': len(review_rows)},
            {'Metric': 'Pairs reviewed: Real overlap', 'Value': sum(1 for r in review_rows if r['Review_Decision'] == 'Real overlap')},
            {'Metric': 'Pairs reviewed: False positive', 'Value': sum(1 for r in review_rows if r['Review_Decision'] == 'False positive')},
            {'Metric': 'Pairs unreviewed', 'Value': sum(1 for r in review_rows if r['Review_Decision'] == 'Unreviewed')},
            {'Metric': 'Suggested removals (duplicates)', 'Value': len(removal_rows)},
        ]
        for k, v in sharp_counts.items():
            summary_rows.append({'Metric': f"Sharpness = {k}", 'Value': v})

        try:
            with pd.ExcelWriter(out_path, engine='openpyxl') as xl:
                pd.DataFrame(label_rows).to_excel(xl, sheet_name="Labels", index=False)
                pd.DataFrame(review_rows if review_rows else
                             [{'Image_A': '', 'Image_B': '', 'IoU_%': '', 'Containment_%': '',
                               'Inliers': '', 'Auto_FP_Flag': '', 'Review_Decision': ''}]
                             ).to_excel(xl, sheet_name="Overlap_Review", index=False)
                pd.DataFrame(removal_rows if removal_rows else
                             [{'Keep': '', 'Suggested_Remove': '', 'Reason': 'none'}]
                             ).to_excel(xl, sheet_name="Removal_Suggestions", index=False)
                pd.DataFrame(summary_rows).to_excel(xl, sheet_name="Summary", index=False)
            messagebox.showinfo("Report Saved", f"Report saved to:\n{os.path.abspath(out_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save report: {e}")

    def _reset_ui(self):
        for w in (self.process_btn, self.browse_main_btn, self.main_folder_entry,
                  self.report_entry, self.browse_report_btn, self.image_folder_entry,
                  self.browse_img_btn, self.start_btn):
            w.config(state=tk.NORMAL)
        self.main_folder_var.set("")
        self.report_excel_var.set("")
        self.image_folder_var.set("")
        self._set_labeling_active(False)
        for w in (self.zoom_out_btn, self.zoom_reset_btn, self.zoom_in_btn):
            w.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.current_pil_image = None
        self.caution_label.config(text="")
        for w in self.overlap_review_frame.winfo_children():
            w.destroy()
        self.sorted_image_info_list = []
        self.current_image_index = 0
        self.overlap_by_image, self.overlap_reviews = {}, {}
        self.sharpness_label_var.set("")
        self.identifier_label_var.set("")
        for v in self.anomaly_vars.values():
            v.set(False)
        self.progress.config(value=0)
        self._status("Process a new folder (Stage 1) or load a report (Stage 2).")

    # ---------------------------------------------------------- misc
    def _status(self, msg):
        self.image_info_label.config(text=f"Status: {msg}")
        self.root.update_idletasks()


if __name__ == '__main__':
    root = tk.Tk()
    try:
        ttk.Style().theme_use('clam')
    except Exception:
        pass
    app = DatasetCleaningApp(root)
    root.mainloop()
