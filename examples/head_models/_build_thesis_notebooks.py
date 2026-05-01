"""Generate thesis Results chapter notebooks 51--57.

Running this script writes seven .ipynb files alongside it. They share the
helpers in `_thesis_helpers.py` and collectively produce the tables and
figures referenced in the Results chapter of the thesis.

This script is idempotent: re-running it overwrites the notebooks. Any
manual edits made inside a notebook will be lost on re-run, so edit the
source cells here, not in the notebook directly.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(*lines: str) -> dict:
    """Build a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(lines),
    }


def code(*lines: str) -> dict:
    """Build a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_lines(lines),
    }


def _to_lines(blocks: tuple[str, ...]) -> list[str]:
    text = "\n\n".join(b.rstrip() for b in blocks)
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    """Wrap a list of cells into a Jupyter notebook dict."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.name}")


COMMON_HEADER = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))
from _thesis_helpers import (
    SUBJECTS, subject_paths, load_raw, load_anon, load_landmarks,
    available_subjects, missing_report, run_pipeline,
)

import numpy as np
import pandas as pd
import xarray as xr

OUT_DIR = pathlib.Path('thesis_results_out')
OUT_DIR.mkdir(exist_ok=True)"""


# ---------------------------------------------------------------------------
# 51 Batch validation
# ---------------------------------------------------------------------------

def nb_51_batch_validation() -> dict:
    return notebook([
        md(
            "# 51 Batch mesh statistics",
            "Runs the anonymization pipeline on every valid thesis subject "
            "and collects the mesh-level quantities that Chapter 4 actually "
            "needs: vertex and face counts before and after deletion, the "
            "percentage of vertices removed, the cap-boundary height, and "
            "the degenerate-face percentage on the anonymized mesh.",
            "The landmark-to-surface distance check that was here earlier "
            "was dropped: the five 10-20 landmarks are stored in the "
            "`_landmarks.tsv` sidecar and are therefore preserved by "
            "construction (the deletion operator does not touch the "
            "landmark array). Optode preservation is the science-relevant "
            "question and is handled in notebook 55. Face-detectability is "
            "handled in notebook 56.",
            "Output: `thesis_results_out/batch_validation.csv`, which "
            "populates the mesh-statistics table and the mesh-integrity "
            "prose of Chapter 4.",
            "**Prerequisite.** Each subject needs a "
            "`Subject{N}_anon_landmarks.tsv` sidecar, written by notebook "
            "48. Subjects without the sidecar are skipped with a warning.",
        ),
        code(COMMON_HEADER),
        md("## 1. Which subjects are ready?"),
        code(
            "ready = available_subjects()",
            "print(f'Ready: {ready}')",
            "missing = missing_report()",
            "if missing:",
            "    print('Missing files:')",
            "    for n, items in missing.items():",
            "        print(f'  Subject{n}: {items}')",
        ),
        md(
            "## 2. Run pipeline per subject, collect mesh statistics",
            "For each ready subject we load the raw scan and landmarks, "
            "run the pipeline, and record mesh statistics on the "
            "CTF-aligned head (`surface_ctf`) and the anonymized mesh "
            "(`surface_anon_ctf`). No validator call; just direct counts.",
        ),
        code(
            "rows = []",
            "for n in ready:",
            "    print(f'--- Subject{n} ---')",
            "    surface_raw = load_raw(n)",
            "    landmarks_raw = load_landmarks(n)",
            "    art = run_pipeline(surface_raw, landmarks_raw, subject=n)",
            "",
            "    n_head = art.surface_ctf.nvertices",
            "    n_faces_head = art.surface_ctf.nfaces",
            "    n_anon = art.surface_anon_ctf.nvertices",
            "    n_faces_anon = art.surface_anon_ctf.nfaces",
            "    mask_size = int(art.mask.sum())",
            "    degen_area = (art.surface_anon_ctf.mesh.area_faces < 1e-12)",
            "    degen_pct = 100.0 * float(degen_area.sum()) / max(1, len(degen_area))",
            "    pct_removed = 100.0 * (n_head - n_anon) / max(1, n_head)",
            "",
            "    row = {",
            "        'subject': n,",
            "        'n_vertices_raw': surface_raw.nvertices,",
            "        'n_faces_raw': surface_raw.nfaces,",
            "        'n_vertices_head': n_head,",
            "        'n_faces_head': n_faces_head,",
            "        'mask_size': mask_size,",
            "        'n_vertices_anonymized': n_anon,",
            "        'n_faces_anonymized': n_faces_anon,",
            "        'vertices_removed': int(n_head - n_anon),",
            "        'faces_removed': int(n_faces_head - n_faces_anon),",
            "        'pct_vertices_removed': pct_removed,",
            "        'degenerate_face_pct': degen_pct,",
            "        'cap_z_mm': art.cap_z,",
            "    }",
            "    rows.append(row)",
            "    print(",
            "        f'  head: {n_head:,} v / {n_faces_head:,} f  ->  '",
            "        f'anon: {n_anon:,} v / {n_faces_anon:,} f  '",
            "        f'(-{pct_removed:.1f}%, degen {degen_pct:.3f}%, '",
            "        f'cap_z {art.cap_z:.1f} mm)'",
            "    )",
        ),
        md(
            "## 3. Summary table",
            "One row per subject. Columns feed the mesh-statistics table "
            "and the mesh-integrity prose of Chapter 4.",
        ),
        code(
            "df = pd.DataFrame(rows)",
            "if len(df):",
            "    df = df.sort_values('subject').reset_index(drop=True)",
            "df",
        ),
        md(
            "## 4. Cohort-level numbers for the chapter prose",
            "The mesh-integrity section cites cohort-level min/max/median "
            "degenerate-face percentage; the mesh-statistics section cites "
            "the overall vertex-removal range. Compute both here so the "
            "thesis text can just quote them.",
        ),
        code(
            "if len(df):",
            "    print(f'pct_vertices_removed: min={df.pct_vertices_removed.min():.2f}, '",
            "          f'median={df.pct_vertices_removed.median():.2f}, '",
            "          f'max={df.pct_vertices_removed.max():.2f}')",
            "    print(f'degenerate_face_pct: min={df.degenerate_face_pct.min():.4f}, '",
            "          f'median={df.degenerate_face_pct.median():.4f}, '",
            "          f'max={df.degenerate_face_pct.max():.4f}')",
            "    print(f'cap_z_mm: min={df.cap_z_mm.min():.1f}, '",
            "          f'median={df.cap_z_mm.median():.1f}, '",
            "          f'max={df.cap_z_mm.max():.1f}')",
        ),
        md("## 5. Save CSV"),
        code(
            "out = OUT_DIR / 'batch_validation.csv'",
            "df.to_csv(out, index=False)",
            "print(f'Wrote {out} ({len(df)} rows)')",
        ),
    ])


# ---------------------------------------------------------------------------
# 52 Pairwise landmark distances
# ---------------------------------------------------------------------------

def nb_52_pairwise_distances() -> dict:
    return notebook([
        md(
            "# 52 Pairwise inter-landmark distances",
            "Deletion preserves every non-masked vertex bit-exact, so the "
            "ten pairwise distances among the five 10-20 landmarks must "
            "match between the original mesh and the anonymized mesh. This "
            "notebook computes that table for one representative subject "
            "(Subject 17 by default) and populates Table 4.3 of the thesis.",
            "The check is reported for one subject because the deletion "
            "operator is the same for every subject; the remaining six "
            "subjects' entries in Table 4.3 would be identical zeros and "
            "add no information.",
        ),
        code(COMMON_HEADER, "SUBJECT = 17"),
        md(
            "## 1. Load original and anonymized landmarks",
            "The TSV written by `save_anonymized_scan` uses the digitized "
            "frame, the same frame as the raw OBJ. Both the original and "
            "anonymized landmarks therefore live in a common coordinate "
            "system; any difference between pairwise distances reflects "
            "landmark displacement, not a frame mismatch.",
        ),
        code(
            "import cedalion.io",
            "paths = subject_paths(SUBJECT)",
            "landmarks_orig = cedalion.io.load_tsv(str(paths.landmarks_tsv))",
            "print('Original landmarks:')",
            "print(landmarks_orig)",
        ),
        md(
            "## 2. Re-run pipeline to obtain the anonymized landmarks",
            "`save_anonymized_scan` writes a single `_landmarks.tsv` -- the "
            "landmarks as they sit on the anonymized output. For this "
            "comparison we want both: the landmarks on the original (just "
            "loaded) and the landmarks after the affine round-trip.  Since "
            "the deletion step itself does not touch the landmark array, "
            "we can simply re-run the pipeline and take "
            "`landmarks_dig`.",
        ),
        code(
            "surface_raw = load_raw(SUBJECT)",
            "art = run_pipeline(surface_raw, landmarks_orig, subject=SUBJECT)",
            "landmarks_anon = art.landmarks_dig",
            "print('Anonymized landmarks:')",
            "print(landmarks_anon)",
        ),
        md(
            "## 3. Compute pairwise distances",
            "All 10 unordered pairs among {Nz, Iz, Cz, Lpa, Rpa}.",
        ),
        code(
            "from itertools import combinations",
            "",
            "def _pos(da, label):",
            "    arr = da.pint.dequantify().values",
            "    labels = list(da['label'].values)",
            "    return arr[labels.index(label)]",
            "",
            "labels = ['Nz', 'Iz', 'Cz', 'LPA', 'RPA']",
            "rows = []",
            "for a, b in combinations(labels, 2):",
            "    d_orig = float(np.linalg.norm(_pos(landmarks_orig, a) - _pos(landmarks_orig, b)))",
            "    d_anon = float(np.linalg.norm(_pos(landmarks_anon, a) - _pos(landmarks_anon, b)))",
            "    rows.append({",
            "        'pair': f'{a}-{b}',",
            "        'd_original_mm': d_orig,",
            "        'd_anonymized_mm': d_anon,",
            "        'abs_diff_mm': abs(d_orig - d_anon),",
            "    })",
            "",
            "df = pd.DataFrame(rows)",
            "df",
        ),
        md("## 4. Save"),
        code(
            "out = OUT_DIR / 'pairwise_distances.csv'",
            "df.to_csv(out, index=False)",
            "print(f'Max |diff|: {df.abs_diff_mm.max():.6g} mm')",
            "print(f'Wrote {out}')",
        ),
    ])


# ---------------------------------------------------------------------------
# 54 Before/after renders
# ---------------------------------------------------------------------------

def nb_54_renders() -> dict:
    return notebook([
        md(
            "# 54 Before/after mesh renders",
            "Loads the representative subject's original and anonymized "
            "meshes side by side in an interactive PyVista viewer with "
            "linked cameras, so rotating one side rotates the other. "
            "Use this notebook to position the camera at the two angles "
            "referenced by the thesis hero figure (frontal and "
            "three-quarter) and export screenshots of each mesh at the "
            "current camera.",
            "Targets for `thesis_results_out/`:",
            "- `hero_original_frontal.png`, `hero_anon_frontal.png`",
            "- `hero_original_threequarter.png`, `hero_anon_threequarter.png`",
            "Data-sharing constraint: only one representative subject "
            "appears in rendered figures; numerical tables cover all seven.",
        ),
        code(
            COMMON_HEADER,
            "import pyvista as pv",
            "from cedalion.vtktutils import trimesh_to_vtk_polydata",
            "",
            "pv.set_jupyter_backend('server')",
            "",
            "HERO_SUBJECT = 17",
            "GREY = 'lightgrey'",
        ),
        md(
            "## 1. Load both meshes",
            "`load_raw` returns the original Einstar scan; `load_anon` "
            "returns the anonymized scan saved by notebook 48. Both are "
            "in the digitized (raw-scanner) frame, so the camera you set "
            "in the viewer applies identically to both.",
        ),
        code(
            "surface_orig = load_raw(HERO_SUBJECT)",
            "surface_anon = load_anon(HERO_SUBJECT)",
            "",
            "poly_orig = pv.wrap(trimesh_to_vtk_polydata(surface_orig.mesh))",
            "poly_anon = pv.wrap(trimesh_to_vtk_polydata(surface_anon.mesh))",
            "print(f'original: {poly_orig.n_points:,} verts, {poly_orig.n_cells:,} faces')",
            "print(f'anon:     {poly_anon.n_points:,} verts, {poly_anon.n_cells:,} faces')",
        ),
        md(
            "## 2. Side-by-side viewer (linked cameras)",
            "Left panel is the original; right panel is the anonymized. "
            "Cameras are linked, so orbiting either side orbits both. "
            "Pick your angle here, then move on to the save cells below "
            "to screenshot each mesh individually at the current view.",
        ),
        code(
            "plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 900), notebook=True)",
            "",
            "plotter.subplot(0, 0)",
            "plotter.add_text('Original', font_size=12)",
            "plotter.add_mesh(poly_orig, color=GREY, smooth_shading=True)",
            "plotter.set_background('white')",
            "",
            "plotter.subplot(0, 1)",
            "plotter.add_text('Anonymized', font_size=12)",
            "plotter.add_mesh(poly_anon, color=GREY, smooth_shading=True)",
            "plotter.set_background('white')",
            "",
            "plotter.link_views()",
            "plotter.show()",
        ),
        md(
            "## 3. Save helper",
            "Run this cell once per angle. It grabs the current camera "
            "from the side-by-side viewer above and renders each mesh "
            "*alone* (full frame, no subplot) at that camera into its "
            "own PNG, so the two hero images can be placed side by side "
            "in the thesis figure without any subplot chrome.",
            "Set `ANGLE` to either `'frontal'` or `'threequarter'` "
            "before running.",
        ),
        code(
            "ANGLE = 'frontal'  # or 'threequarter'",
            "WINDOW = (1200, 1600)",
            "",
            "cam = plotter.camera_position",
            "print('camera_position =', cam)",
            "",
            "for tag, poly in (('original', poly_orig), ('anon', poly_anon)):",
            "    p = pv.Plotter(off_screen=True, window_size=WINDOW)",
            "    p.add_mesh(poly, color=GREY, smooth_shading=True)",
            "    p.set_background('white')",
            "    p.enable_anti_aliasing('ssaa')",
            "    p.camera_position = cam",
            "    out = OUT_DIR / f'hero_{tag}_{ANGLE}.png'",
            "    p.screenshot(str(out))",
            "    p.close()",
            "    print(f'wrote {out}')",
        ),
    ])


# ---------------------------------------------------------------------------
# 55 Optode co-registration invariance
# ---------------------------------------------------------------------------

def nb_55_coreg_invariance() -> dict:
    return notebook([
        md(
            "# 55 Optode co-registration invariance",
            "For each subject, run `ColoredStickerProcessor` independently "
            "on the original and on the anonymized mesh, match the "
            "detected sticker centres between the two runs by nearest "
            "neighbour, and report per-optode Euclidean deviations for "
            "both the raw sticker centres and the scalp-projected "
            "optode positions (sticker centre minus optode length along "
            "the surface normal).",
            "Populates Table 4.5 of the thesis. Expected outcome: zero up "
            "to numerical noise, because every sticker sits on the cap "
            "which lies outside the facial mask by construction.",
        ),
        code(
            COMMON_HEADER,
            "import cedalion",
            "from cedalion.geometry.photogrammetry.processors import (",
            "    ColoredStickerProcessor,",
            ")",
            "from scipy.spatial import KDTree",
            "",
            "# Yellow optode stickers, same HSV range as notebook 41.",
            "COLORS = {'O': (0.11, 0.21, 0.7, 1.0)}",
            "OPTODE_LENGTH = 22.6 * cedalion.units.mm",
        ),
        md(
            "## 1. Sticker detection wrapper",
            "Returns sticker centres and normals as plain numpy arrays "
            "for easy nearest-neighbour matching.",
        ),
        code(
            "def detect_stickers(surface):",
            "    proc = ColoredStickerProcessor(colors=COLORS)",
            "    centres, normals = proc.process(surface, details=False)",
            "    c_np = centres.pint.dequantify().values",
            "    n_np = normals.values if hasattr(normals, 'values') else np.asarray(normals)",
            "    return c_np, n_np",
            "",
            "def match_by_nn(a, b):",
            "    '''For each row of a, return the index of the nearest row of b.'''",
            "    tree = KDTree(b)",
            "    _, idx = tree.query(a)",
            "    return idx",
        ),
        md("## 2. Per-subject comparison"),
        code(
            "rows = []",
            "for n in SUBJECTS:",
            "    paths = subject_paths(n)",
            "    if not (paths.raw_exists and paths.anon_exists):",
            "        print(f'skipping Subject{n}: missing scans')",
            "        continue",
            "    print(f'--- Subject{n} ---')",
            "    surface_orig = load_raw(n)",
            "    surface_anon = load_anon(n)",
            "",
            "    c_orig, n_orig = detect_stickers(surface_orig)",
            "    c_anon, n_anon = detect_stickers(surface_anon)",
            "",
            "    if len(c_orig) == 0 or len(c_anon) == 0:",
            "        print(f'  no stickers detected on one side '",
            "              f'(orig={len(c_orig)}, anon={len(c_anon)})')",
            "        continue",
            "",
            "    idx = match_by_nn(c_orig, c_anon)",
            "    sticker_dev = np.linalg.norm(c_orig - c_anon[idx], axis=1)",
            "",
            "    L = float(OPTODE_LENGTH.to('mm').magnitude)",
            "    scalp_orig = c_orig - L * n_orig",
            "    scalp_anon = c_anon[idx] - L * n_anon[idx]",
            "    scalp_dev = np.linalg.norm(scalp_orig - scalp_anon, axis=1)",
            "",
            "    rows.append({",
            "        'subject': n,",
            "        'n_stickers_original': len(c_orig),",
            "        'n_stickers_anonymized': len(c_anon),",
            "        'n_matched': len(idx),",
            "        'sticker_mean_mm': float(sticker_dev.mean()),",
            "        'sticker_median_mm': float(np.median(sticker_dev)),",
            "        'sticker_max_mm': float(sticker_dev.max()),",
            "        'scalp_mean_mm': float(scalp_dev.mean()),",
            "        'scalp_median_mm': float(np.median(scalp_dev)),",
            "        'scalp_max_mm': float(scalp_dev.max()),",
            "    })",
            "    print(rows[-1])",
        ),
        md("## 3. Summary"),
        code(
            "df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True)",
            "df",
        ),
        md("## 4. Save"),
        code(
            "out = OUT_DIR / 'coreg_invariance.csv'",
            "df.to_csv(out, index=False)",
            "print(f'Wrote {out}')",
        ),
    ])


# ---------------------------------------------------------------------------
# 56 Face-detectability
# ---------------------------------------------------------------------------

def nb_56_face_detectability() -> dict:
    return notebook([
        md(
            "# 56 Face-detectability check",
            "Render the anonymized mesh from a sweep of viewpoints and "
            "run the MediaPipe Face Detector on every view. For vertex "
            "deletion the expected outcome is zero detections: there is "
            "no face surface left for the detector to latch onto. "
            "Detection counts are reported for every subject "
            "(Table 4.7); the contact-sheet figure is rendered for "
            "Subject 17 only, because thesis data-sharing rules allow "
            "identifiable renders for Subject 17 only.",
            "Populates Figure 4.6 (contact sheet) and Table 4.7 "
            "(per-subject detection counts) of the thesis.",
            "Dependency: `mediapipe`. Install into the cedalion env with "
            "`pip install mediapipe` if not yet present.",
        ),
        code(
            COMMON_HEADER,
            "import pyvista as pv",
            "pv.OFF_SCREEN = True",
            "import cv2  # for BGR -> RGB conversion",
            "",
            "# Yaw/pitch sweep in degrees",
            "YAWS = list(range(-90, 91, 30))   # -90 .. 90 step 30",
            "PITCHES = [-20, 0, 20]",
            "WINDOW = (640, 640)",
            "GREY = (200, 200, 200)",
        ),
        md(
            "## 1. Render sweep",
            "Rotate the camera around the anonymized head in yaw/pitch, "
            "save each frame to disk so MediaPipe can read it.",
        ),
        code(
            "from cedalion.vtktutils import trimesh_to_vtk_polydata",
            "",
            "def render_sweep(surface, out_dir, subject_n):",
            "    out_dir.mkdir(parents=True, exist_ok=True)",
            "    poly = pv.wrap(trimesh_to_vtk_polydata(surface.mesh))",
            "    files = []",
            "    for yaw in YAWS:",
            "        for pitch in PITCHES:",
            "            pvplt = pv.Plotter(off_screen=True, window_size=WINDOW)",
            "            pvplt.add_mesh(poly, color=[c/255 for c in GREY], smooth_shading=True)",
            "            pvplt.set_background('white')",
            "            pvplt.enable_anti_aliasing('ssaa')",
            "            pvplt.view_xz()",
            "            pvplt.camera.azimuth = 180 + yaw",
            "            pvplt.camera.elevation = pitch",
            "            pvplt.camera.zoom(1.3)",
            "            fn = out_dir / f'subject{subject_n}_yaw{yaw:+04d}_pitch{pitch:+03d}.png'",
            "            pvplt.screenshot(str(fn))",
            "            pvplt.close()",
            "            files.append((yaw, pitch, fn))",
            "    return files",
        ),
        md(
            "## 2. MediaPipe detector",
            "Uses the short-range Face Detection model. Confidence "
            "threshold is left at the default 0.5; we also record the "
            "maximum confidence observed per view for the table.",
        ),
        code(
            "import mediapipe as mp",
            "mp_face_detection = mp.solutions.face_detection",
            "",
            "def detect_faces(image_path):",
            "    img = cv2.imread(str(image_path))",
            "    if img is None:",
            "        return 0, 0.0",
            "    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)",
            "    with mp_face_detection.FaceDetection(",
            "        model_selection=0, min_detection_confidence=0.5",
            "    ) as fd:",
            "        result = fd.process(rgb)",
            "    if result.detections is None:",
            "        return 0, 0.0",
            "    confidences = [d.score[0] for d in result.detections]",
            "    return len(confidences), float(max(confidences))",
        ),
        md("## 3. Per-subject sweep + detection"),
        code(
            "view_root = OUT_DIR / 'detectability_views'",
            "rows = []",
            "for n in SUBJECTS:",
            "    if not subject_paths(n).anon_exists:",
            "        print(f'skipping Subject{n}: no anon .obj')",
            "        continue",
            "    print(f'--- Subject{n} ---')",
            "    surface = load_anon(n)",
            "    files = render_sweep(surface, view_root / f'subject{n}', n)",
            "    hits = 0",
            "    max_conf = 0.0",
            "    n_views = len(files)",
            "    for yaw, pitch, fn in files:",
            "        k, c = detect_faces(fn)",
            "        hits += int(k > 0)",
            "        if c > max_conf:",
            "            max_conf = c",
            "    rows.append({",
            "        'subject': n,",
            "        'n_views': n_views,",
            "        'detector_hits': hits,",
            "        'max_confidence': max_conf,",
            "    })",
            "    print(rows[-1])",
        ),
        md("## 4. Summary + contact sheet"),
        code(
            "df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True)",
            "df",
        ),
        code(
            "import matplotlib.pyplot as plt",
            "import matplotlib.image as mpimg",
            "",
            "# Thesis rule: only Subject 17 may appear in a rendered figure.",
            "CONTACT_SUBJECT = 17 if 17 in list(df.subject) else None",
            "if CONTACT_SUBJECT is not None:",
            "    dir_ = view_root / f'subject{CONTACT_SUBJECT}'",
            "    imgs = sorted(dir_.glob('*.png'))",
            "    n_cols = 7",
            "    n_rows = int(np.ceil(len(imgs) / n_cols))",
            "    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4*n_cols, 2.4*n_rows))",
            "    for ax, img_path in zip(np.atleast_1d(axes).ravel(), imgs):",
            "        ax.imshow(mpimg.imread(img_path))",
            "        ax.set_xticks([]); ax.set_yticks([])",
            "        ax.set_title(img_path.stem.split('_', 1)[1], fontsize=6)",
            "    for ax in np.atleast_1d(axes).ravel()[len(imgs):]:",
            "        ax.axis('off')",
            "    fig.suptitle(f'Face detectability contact sheet - Subject{CONTACT_SUBJECT}')",
            "    fig.tight_layout()",
            "    out = OUT_DIR / 'detectability_contact.pdf'",
            "    fig.savefig(out, dpi=200, bbox_inches='tight')",
            "    print(f'Wrote {out}')",
        ),
        code(
            "out_csv = OUT_DIR / 'detectability_summary.csv'",
            "df.to_csv(out_csv, index=False)",
            "print(f'Wrote {out_csv}')",
        ),
    ])


# ---------------------------------------------------------------------------
# 57 Auxiliary MediaPipe nasion
# ---------------------------------------------------------------------------

def nb_57_auxiliary_nasion() -> dict:
    return notebook([
        md(
            "# 57 Auxiliary MediaPipe nasion route",
            "Compares the automatic MediaPipe nasion detector to the "
            "manually picked nasion across the seven thesis subjects. "
            "Feeds Table 4.8 (confidence + distance) and Figure 4.7 "
            "(midline Y-profile plot for Subject 17) of the thesis.",
            "**BRANCH REQUIRED.** The auto-nasion code was moved out of "
            "`main` in commit 95a7a4c and now lives on the "
            "`auto-detection-pipeline` branch. Before running this "
            "notebook:",
            "```",
            "cd /home/ma7/BA/cedalion/cedalion",
            "git checkout auto-detection-pipeline",
            "```",
            "Then run this notebook. Remember to switch back to `main` "
            "afterwards so the other Results notebooks keep working.",
        ),
        code(
            COMMON_HEADER,
            "import matplotlib.pyplot as plt",
        ),
        md(
            "## 1. Import the auto-nasion detector",
            "This import only succeeds on the `auto-detection-pipeline` "
            "branch. On `main` it raises `ImportError` and the notebook "
            "cleanly reports that the branch switch is required.",
        ),
        code(
            "try:",
            "    from cedalion.geometry.photogrammetry.anonymization.nasion_detector import (",
            "        detect_nasion_auto,",
            "    )",
            "    HAS_AUTO = True",
            "except ImportError as err:",
            "    HAS_AUTO = False",
            "    print('Auto-nasion detector is not on this branch:')",
            "    print(f'  {err}')",
            "    print('Run: git checkout auto-detection-pipeline')",
        ),
        md("## 2. Per-subject comparison"),
        code(
            "rows = []",
            "profiles = {}  # subject -> midline Y-profile for plotting",
            "if HAS_AUTO:",
            "    for n in SUBJECTS:",
            "        paths = subject_paths(n)",
            "        if not (paths.raw_exists and paths.landmarks_exist):",
            "            print(f'skipping Subject{n}: missing raw or landmarks')",
            "            continue",
            "        surface_raw = load_raw(n)",
            "        landmarks_raw = load_landmarks(n)",
            "",
            "        Nz_manual = landmarks_raw.sel(label='Nz').pint.dequantify().values",
            "",
            "        result = detect_nasion_auto(surface_raw)",
            "        Nz_auto = result.position",
            "        conf = float(result.confidence)",
            "        dist = float(np.linalg.norm(Nz_manual - Nz_auto))",
            "        outcome = 'success' if conf >= 0.3 else 'fallback'",
            "",
            "        rows.append({",
            "            'subject': n,",
            "            'confidence': conf,",
            "            'distance_to_manual_mm': dist,",
            "            'outcome': outcome,",
            "        })",
            "        # Keep the midline profile if the detector exposes it",
            "        profiles[n] = getattr(result, 'profile', None)",
            "        print(rows[-1])",
        ),
        md("## 3. Summary table"),
        code(
            "df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True) if rows else pd.DataFrame()",
            "df",
        ),
        md(
            "## 4. Figure 4.7: midline Y-profile for Subject 17",
            "Plots the midline Y(z) profile with the auto-detected and "
            "manually picked nasion annotated. The figure is scoped to "
            "Subject 17 only because thesis data-sharing rules restrict "
            "subject-identifiable figures to Subject 17, even though the "
            "numeric table (Table 4.8) covers all seven subjects.",
        ),
        code(
            "FIG_SUBJECT = 17",
            "if FIG_SUBJECT in profiles and profiles[FIG_SUBJECT] is not None:",
            "    prof = profiles[FIG_SUBJECT]",
            "    r = df[df.subject == FIG_SUBJECT].iloc[0]",
            "    fig, ax = plt.subplots(figsize=(8, 4))",
            "    ax.plot(prof['z'], prof['y'], label='midline Y(z)')",
            "    ax.set_title(f'Subject{FIG_SUBJECT} (conf={r.confidence:.2f}, "
            "d={r.distance_to_manual_mm:.1f} mm)')",
            "    ax.set_xlabel('Z (mm)')",
            "    ax.set_ylabel('Y (mm)')",
            "    ax.legend()",
            "    fig.tight_layout()",
            "    out = OUT_DIR / 'auxiliary_nasion_profile.pdf'",
            "    fig.savefig(out, dpi=200, bbox_inches='tight')",
            "    print(f'Wrote {out}')",
            "else:",
            "    print(f'No profile for Subject{FIG_SUBJECT}; skipping figure.')",
        ),
        md("## 5. Save CSV"),
        code(
            "if len(df):",
            "    out = OUT_DIR / 'auxiliary_nasion.csv'",
            "    df.to_csv(out, index=False)",
            "    print(f'Wrote {out}')",
        ),
    ])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "51_batch_validation.ipynb": nb_51_batch_validation,
    "52_pairwise_distances.ipynb": nb_52_pairwise_distances,
    "54_before_after_renders.ipynb": nb_54_renders,
    "55_coreg_invariance.ipynb": nb_55_coreg_invariance,
    "56_face_detectability.ipynb": nb_56_face_detectability,
    "57_auxiliary_nasion.ipynb": nb_57_auxiliary_nasion,
}


def main() -> None:
    for fname, factory in NOTEBOOKS.items():
        write(HERE / fname, factory())


if __name__ == "__main__":
    main()
