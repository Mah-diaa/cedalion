"""Generate the CSV-producing thesis notebooks (64, 68, 70).

Running this script writes three .ipynb files alongside it. Each
notebook imports cohort/IO helpers from `_thesis_data.py` and the
pipeline wrapper from `_thesis_pipeline.py`, then calls the public
anonymization API (and `validate_anonymization` for nb 64) to produce
the CSV tables cited in the Results chapter.

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
from _thesis_data import (
    SUBJECTS, subject_paths, load_raw, load_anon, load_landmarks,
    available_subjects, missing_report,
)
from _thesis_pipeline import run_pipeline

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
            "Structural checks (vertex-count delta, mesh validity, "
            "degenerate-face ratio, protected-point preservation) come "
            "from the shipped `validate_anonymization` -- the same "
            "function the production pipeline could invoke -- so the "
            "thesis numbers are produced by the function the codebase "
            "actually exports, not a notebook re-implementation.",
            "Output: `thesis_results_out/batch_validation.csv`, which "
            "populates the mesh-statistics table and the mesh-integrity "
            "prose of Chapter 4.",
            "**Prerequisite.** Each subject needs a "
            "`Subject{N}_anon_landmarks.tsv` sidecar, written by notebook "
            "48. Subjects without the sidecar are skipped with a warning.",
        ),
        code(
            COMMON_HEADER,
            "from cedalion.geometry.photogrammetry.anonymization import (",
            "    validate_anonymization,",
            ")",
        ),
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
            "run the pipeline (via the wrapper that exposes intermediates) "
            "and pass the CTF-aligned head, the anonymized mesh, the "
            "deletion mask, and the landmarks to "
            "`validate_anonymization`. The wrapper hands us "
            "`pct_vertices_removed` and `cap_z_mm` directly; everything "
            "else comes from the validator.",
        ),
        code(
            "rows = []",
            "for n in ready:",
            "    print(f'--- Subject{n} ---')",
            "    surface_raw = load_raw(n)",
            "    landmarks_raw = load_landmarks(n)",
            "    art = run_pipeline(surface_raw, landmarks_raw, subject=n)",
            "",
            "    result = validate_anonymization(",
            "        original_surface=art.surface_ctf,",
            "        anonymized_surface=art.surface_anon_ctf,",
            "        facial_mask=art.mask,",
            "        protected_points=art.landmarks_ctf,",
            "    )",
            "",
            "    n_head = art.surface_ctf.nvertices",
            "    pct_removed = 100.0 * result.actual_vertices_removed / max(1, n_head)",
            "",
            "    row = {",
            "        'subject': n,",
            "        'n_vertices_raw': surface_raw.nvertices,",
            "        'n_faces_raw': surface_raw.nfaces,",
            "        'n_vertices_head': n_head,",
            "        'n_faces_head': art.surface_ctf.nfaces,",
            "        'mask_size': result.expected_vertices_removed,",
            "        'n_vertices_anonymized': art.surface_anon_ctf.nvertices,",
            "        'n_faces_anonymized': art.surface_anon_ctf.nfaces,",
            "        'vertices_removed': result.actual_vertices_removed,",
            "        'faces_removed': int(art.surface_ctf.nfaces - art.surface_anon_ctf.nfaces),",
            "        'pct_vertices_removed': pct_removed,",
            "        'degenerate_face_pct': result.degenerate_face_pct,",
            "        'protected_point_max_delta_mm': result.protected_point_max_delta_mm,",
            "        'cap_z_mm': art.cap_z,",
            "        'passed': result.passed,",
            "    }",
            "    rows.append(row)",
            "    print(",
            "        f'  head: {n_head:,} v / {art.surface_ctf.nfaces:,} f  ->  '",
            "        f'anon: {art.surface_anon_ctf.nvertices:,} v / "
            "{art.surface_anon_ctf.nfaces:,} f  '",
            "        f'(-{pct_removed:.1f}%, degen {result.degenerate_face_pct:.3f}%, '",
            "        f'cap_z {art.cap_z:.1f} mm)  {result.summary}'",
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
            "from _thesis_data import OPTODE_SUBJECTS",
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
    "68_coreg_invariance.ipynb": nb_68_coreg_invariance,
    "70_auxiliary_nasion.ipynb": nb_70_auxiliary_nasion,
}


def main() -> None:
    for fname, factory in NOTEBOOKS.items():
        write(HERE / fname, factory())


if __name__ == "__main__":
    main()
