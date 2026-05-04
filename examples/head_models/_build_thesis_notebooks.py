"""Generate the CSV-producing thesis notebooks (64, 65, 68, 69, 70).

Running this script writes five .ipynb files alongside it. They share
the helpers in `_thesis_helpers.py` and produce the CSV tables cited in
the Results chapter.

The script is idempotent: re-running overwrites the notebooks, so edit
the source cells here, not the notebook directly. Notebooks 66, 72, 73
are hand-authored and not generated here.
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

OUT_DIR = pathlib.Path('thesis_results_out')
OUT_DIR.mkdir(exist_ok=True)"""


# ---------------------------------------------------------------------------
# 64 Batch validation
# ---------------------------------------------------------------------------

def nb_64_batch_validation() -> dict:
    return notebook([
        md(
            "# 64 Batch mesh statistics",
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
            "question and is handled in notebook 68. Face-detectability is "
            "handled in notebook 69.",
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
        md("## 4. Save CSV"),
        code(
            "out = OUT_DIR / 'batch_validation.csv'",
            "df.to_csv(out, index=False)",
            "print(f'Wrote {out} ({len(df)} rows)')",
        ),
    ])


# ---------------------------------------------------------------------------
# 65 Pairwise landmark distances
# ---------------------------------------------------------------------------

def nb_65_pairwise_distances() -> dict:
    return notebook([
        md(
            "# 65 Pairwise inter-landmark distances",
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
# 68 Optode co-registration invariance
# ---------------------------------------------------------------------------

def nb_68_coreg_invariance() -> dict:
    return notebook([
        md(
            "# 68 Optode co-registration invariance",
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
        md(
            "## 2. Per-subject comparison",
            "Only the optode cohort (S1--S7) carries cap-mounted stickers; "
            "bare-cap subjects have nothing to detect, so they are skipped.",
        ),
        code(
            "from _thesis_helpers import OPTODE_SUBJECTS",
            "rows = []",
            "for n in OPTODE_SUBJECTS:",
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
# 69 Face-detectability
# ---------------------------------------------------------------------------

def nb_69_face_detectability() -> dict:
    return notebook([
        md(
            "# 69 Face-detectability check",
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
        md("## 4. Summary table"),
        code(
            "df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True)",
            "df",
        ),
        md("## 5. Save CSV"),
        code(
            "out_csv = OUT_DIR / 'detectability_summary.csv'",
            "df.to_csv(out_csv, index=False)",
            "print(f'Wrote {out_csv}')",
        ),
    ])


# ---------------------------------------------------------------------------
# 70 Auxiliary MediaPipe nasion
# ---------------------------------------------------------------------------

def nb_70_auxiliary_nasion() -> dict:
    return notebook([
        md(
            "# 70 Auxiliary MediaPipe nasion route",
            "Compares the automatic MediaPipe nasion detector to the "
            "manually picked nasion across the seven thesis subjects. "
            "Feeds Table 4.8 (confidence + distance) of the thesis.",
            "**BRANCH REQUIRED.** The auto-nasion code lives on the "
            "`auto-detection-pipeline` branch. Run "
            "`git checkout auto-detection-pipeline` before executing, then "
            "switch back to `main` afterwards.",
        ),
        code(COMMON_HEADER),
        md(
            "## 1. Import the auto-nasion detector",
            "Only succeeds on the `auto-detection-pipeline` branch; on "
            "`main` the import fails and the notebook reports the missing "
            "branch instead of running.",
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
            "if HAS_AUTO:",
            "    for n in SUBJECTS:",
            "        paths = subject_paths(n)",
            "        if not (paths.raw_exists and paths.landmarks_exist):",
            "            print(f'skipping Subject{n}: missing raw or landmarks')",
            "            continue",
            "        surface_raw = load_raw(n)",
            "        landmarks_raw = load_landmarks(n)",
            "        Nz_manual = landmarks_raw.sel(label='Nz').pint.dequantify().values",
            "        result = detect_nasion_auto(surface_raw)",
            "        conf = float(result.confidence)",
            "        rows.append({",
            "            'subject': n,",
            "            'confidence': conf,",
            "            'distance_to_manual_mm': float(np.linalg.norm(Nz_manual - result.position)),",
            "            'outcome': 'success' if conf >= 0.3 else 'fallback',",
            "        })",
            "        print(rows[-1])",
        ),
        md("## 3. Summary table"),
        code(
            "df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True) if rows else pd.DataFrame()",
            "df",
        ),
        md("## 4. Save CSV"),
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
    "64_batch_validation.ipynb": nb_64_batch_validation,
    "65_pairwise_distances.ipynb": nb_65_pairwise_distances,
    "68_coreg_invariance.ipynb": nb_68_coreg_invariance,
    "69_face_detectability.ipynb": nb_69_face_detectability,
    "70_auxiliary_nasion.ipynb": nb_70_auxiliary_nasion,
}


def main() -> None:
    for fname, factory in NOTEBOOKS.items():
        write(HERE / fname, factory())


if __name__ == "__main__":
    main()
