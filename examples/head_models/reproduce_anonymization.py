"""Batch face-anonymization script for thesis reproducibility.

Discovers subjects whose raw Einstar scan and ``_anon_landmarks.tsv``
sidecar are both present, then re-runs ``anonymize_scan`` on each and
writes the anonymized OBJ + landmark TSV.  Running the same script on the
same inputs always produces the same outputs (no interactive steps).

Scan directory layout expected::

    <SCANS_FOLDER>/Subject<N>/Subject<N>.obj          # raw Einstar scan
    <SCANS_FOLDER>/Subject<N>/Subject<N>_anon_landmarks.tsv  # from notebook 51

Output written to the same folder::

    <SCANS_FOLDER>/Subject<N>/Subject<N>_anon.obj
    <SCANS_FOLDER>/Subject<N>/Subject<N>_anon_landmarks.tsv  (overwritten)

Usage::

    cd examples/head_models
    python reproduce_anonymization.py           # batch anonymize
    python reproduce_anonymization.py --profile # same, plus per-function timing

Prerequisites:
  - Run ``51_manual_5pt_anonymization.ipynb`` for each subject to pick the
    five 10-20 landmarks (Nz, Iz, Cz, LPA, RPA) and write the TSV sidecar.
  - The conda environment must be active (``conda activate cedalion``).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import cedalion.io
from cedalion.geometry.photogrammetry.anonymization import (
    anonymize_scan,
    save_anonymized_scan,
)

# ---------------------------------------------------------------------------
# Edit this constant to point at your scan directory.
# ---------------------------------------------------------------------------
SCANS_FOLDER = Path("./PG_Subjects")
# ---------------------------------------------------------------------------


def _find_subjects(scans_folder: Path) -> list[tuple[int, Path, Path]]:
    """Return (subject_number, raw_obj, landmarks_tsv) for ready subjects."""
    ready = []
    if not scans_folder.exists():
        return ready
    for subj_dir in sorted(scans_folder.iterdir()):
        name = subj_dir.name
        if not (subj_dir.is_dir() and name.startswith("Subject")):
            continue
        try:
            n = int(name[len("Subject"):])
        except ValueError:
            continue
        obj = subj_dir / f"Subject{n}.obj"
        tsv = subj_dir / f"Subject{n}_anon_landmarks.tsv"
        if obj.exists() and tsv.exists():
            ready.append((n, obj, tsv))
    return ready


def _run_one(n: int, obj: Path, tsv: Path) -> tuple[float, list[str]]:
    """Anonymize one subject; return (elapsed_seconds, written_paths)."""
    surface = cedalion.io.read_einstar_obj(str(obj))
    landmarks = cedalion.io.load_tsv(str(tsv))

    t0 = time.perf_counter()
    surface_anon, landmarks_anon = anonymize_scan(surface, landmarks)
    elapsed = time.perf_counter() - t0

    out_obj = str(obj.parent / f"Subject{n}_anon.obj")
    written = save_anonymized_scan(
        surface_anon, out_obj, landmarks=landmarks_anon
    )
    return elapsed, written


def _profile_one(n: int, obj: Path, tsv: Path, top_n: int = 20) -> None:
    """Profile anonymize_scan for one subject and print the hot functions."""
    surface = cedalion.io.read_einstar_obj(str(obj))
    landmarks = cedalion.io.load_tsv(str(tsv))

    pr = cProfile.Profile()
    pr.enable()
    anonymize_scan(surface, landmarks)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(top_n)
    print(buf.getvalue())


def main(profile: bool = False) -> None:
    subjects = _find_subjects(SCANS_FOLDER)
    if not subjects:
        print(
            f"No ready subjects found under {SCANS_FOLDER.resolve()}.\n"
            "Run notebook 51 for each subject first to write the "
            "_anon_landmarks.tsv sidecars."
        )
        return

    print(f"Found {len(subjects)} ready subject(s).\n")

    if profile:
        n, obj, tsv = subjects[0]
        print(f"Profiling Subject{n} (first ready subject)...\n")
        _profile_one(n, obj, tsv)
        return

    total_t0 = time.perf_counter()
    for n, obj, tsv in subjects:
        elapsed, written = _run_one(n, obj, tsv)
        written_names = ", ".join(Path(p).name for p in written)
        print(f"Subject{n:>3}  {elapsed:5.2f}s  -> {written_names}")

    total = time.perf_counter() - total_t0
    print(f"\n{len(subjects)} subjects in {total:.1f}s "
          f"({total / len(subjects):.1f}s/subject)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile anonymize_scan on the first ready subject instead of "
             "running the full batch.",
    )
    args = parser.parse_args()
    main(profile=args.profile)
