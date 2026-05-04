"""Cohort metadata and disk I/O for the thesis CSV-producing notebooks.

The thesis cohort is eleven valid subjects:

- Optode-cap cohort (S1--S7): Subject 16-22 -- worn an fNIRS optode
  cap with cap-mounted optode markers at scan time.
- Bare-cap cohort (S8--S11): Subject 12-15 -- worn a bare cap. Included
  to show the pipeline generalises beyond the optode regime.

Subject 11 was acquired but is excluded (scan-side defect). The optode
co-registration check (notebook 68) detects sticker markers on the cap
and therefore only runs on the optode cohort.

Landmarks come from a `{stem}_landmarks.tsv` sidecar written by
`save_anonymized_scan`. Missing sidecars raise rather than fall back
silently.

This module deliberately holds only cohort constants and disk I/O. The
pipeline wrapper lives in `_thesis_pipeline.py`; rendering and detector
helpers in `_validator_render.py`; the rejected noise operator in
`_validator_noise.py`.

Use this module from a notebook with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path().resolve()))
    from _thesis_data import (
        SUBJECTS, OPTODE_SUBJECTS, BARE_CAP_SUBJECTS,
        is_optode, s_id, load_raw, load_anon, load_landmarks,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.io


OPTODE_SUBJECTS = [16, 17, 18, 19, 20, 21, 22]
BARE_CAP_SUBJECTS = [12, 13, 14, 15]
# Optode subjects come first so S1..S7 maps to Subject 16..22 unchanged from
# the earlier thesis revisions; bare-cap subjects extend the numbering as
# S8..S11 -> Subject 12..15.
SUBJECTS = OPTODE_SUBJECTS + BARE_CAP_SUBJECTS
SCANS_FOLDER = Path("/home/ma7/BA/PG_Subjects")


def is_optode(n: int) -> bool:
    """True if Subject n belongs to the optode-cap cohort."""
    return n in OPTODE_SUBJECTS


def s_id(n: int) -> str:
    """Return the thesis label ("S1".."S11") for Subject n.

    The mapping is the position of `n` in the `SUBJECTS` list, which
    deliberately lists optode-cap subjects first so that S1--S7 keeps
    pointing at Subject 16--22 across thesis revisions.
    """
    return f"S{SUBJECTS.index(n) + 1}"


@dataclass
class SubjectPaths:
    """Canonical file paths for one subject."""

    obj: Path
    anon_obj: Path
    landmarks_tsv: Path

    @property
    def raw_exists(self) -> bool:
        return self.obj.exists()

    @property
    def anon_exists(self) -> bool:
        return self.anon_obj.exists()

    @property
    def landmarks_exist(self) -> bool:
        return self.landmarks_tsv.exists()


def subject_paths(n: int) -> SubjectPaths:
    """Return the canonical file paths for Subject n."""
    folder = SCANS_FOLDER / f"Subject{n}"
    return SubjectPaths(
        obj=folder / f"Subject{n}.obj",
        anon_obj=folder / f"Subject{n}_anon.obj",
        landmarks_tsv=folder / f"Subject{n}_anon_landmarks.tsv",
    )


_RUN_51_HINT = (
    "Run notebook 51 on this subject first; it writes the anonymized "
    "OBJ and the `_landmarks.tsv` sidecar via save_anonymized_scan."
)


def load_raw(n: int) -> cdc.TrimeshSurface:
    """Load the raw Einstar scan for Subject n (digitized frame)."""
    return cedalion.io.read_einstar_obj(str(subject_paths(n).obj))


def load_anon(n: int) -> cdc.TrimeshSurface:
    """Load the anonymized scan for Subject n (digitized frame)."""
    paths = subject_paths(n)
    if not paths.anon_exists:
        raise FileNotFoundError(
            f"No anonymized scan for Subject{n} at {paths.anon_obj}. "
            f"{_RUN_51_HINT}"
        )
    return cedalion.io.read_einstar_obj(str(paths.anon_obj))


def load_landmarks(n: int) -> xr.DataArray:
    """Load the five 10-20 landmarks for Subject n (digitized frame).

    Returns:
        A LabeledPoints DataArray in the digitized frame, same convention
        as `read_einstar_obj` output.
    """
    paths = subject_paths(n)
    if not paths.landmarks_exist:
        raise FileNotFoundError(
            f"No landmarks TSV for Subject{n} at {paths.landmarks_tsv}. "
            f"{_RUN_51_HINT}"
        )
    return cedalion.io.load_tsv(str(paths.landmarks_tsv))


def available_subjects() -> list[int]:
    """Subjects whose raw scan and landmarks TSV both exist."""
    return [
        n for n in SUBJECTS
        if subject_paths(n).raw_exists and subject_paths(n).landmarks_exist
    ]


_FILE_CHECKS = (
    ("raw_exists", "raw .obj"),
    ("landmarks_exist", "landmarks .tsv"),
    ("anon_exists", "anonymized .obj"),
)


def missing_report() -> dict[int, list[str]]:
    """Map Subject n -> list of missing required files (raw, landmarks, anon)."""
    out: dict[int, list[str]] = {}
    for n in SUBJECTS:
        paths = subject_paths(n)
        missing = [label for attr, label in _FILE_CHECKS if not getattr(paths, attr)]
        if missing:
            out[n] = missing
    return out
