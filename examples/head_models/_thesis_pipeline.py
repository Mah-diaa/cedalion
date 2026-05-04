"""Pipeline wrapper that exposes intermediate artifacts.

The shipped `cedalion.geometry.photogrammetry.anonymization.anonymize_scan`
returns just `(surface, landmarks)`. Validation notebooks need richer
state: the deletion mask, the cap-detection profile, both CTF-frame
surfaces, and both inverse transforms. `run_pipeline` reproduces the
canonical orchestration step-for-step using the public anonymization API
and packages every intermediate into a `PipelineArtifacts` dataclass.

Mirror this against
`cedalion.geometry.photogrammetry.anonymization.pipeline.anonymize_scan`
when either side changes; the step ordering and parameters must stay in
lock-step or the validation numbers diverge from the shipped pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion.geometry.landmarks import normalize_landmarks_labels
from cedalion.geometry.photogrammetry.anonymization import (
    align_axes_from_landmarks,
    delete_masked_vertices,
    detect_cap_boundary,
    face_mask_from_landmarks,
    isolate_head,
    normalize_axes,
    revert_to_einstar_frame,
)
from cedalion.geometry.photogrammetry.anonymization._utils import (
    _apply_affine,
    _transform_labeled_points,
)


LANDMARK_KEEP_RADIUS_MM = 8.0
EAR_DELETE_RADIUS_MM = 40.0


@dataclass
class PipelineArtifacts:
    """Every intermediate product of one pipeline run.

    All frames:
    - `*_raw`: digitized (raw Einstar) frame, as read from disk
    - `*_n`: post `normalize_axes`, rotated so +X is up-ish
    - `*_head`: post `isolate_head`, shoulders stripped
    - `*_ctf`: CTF frame (+X anterior, +Y left, +Z up, origin at LPA/RPA midpoint)
    - `*_dig`: reverted back to digitized frame for saving
    """

    subject: int

    # Raw + landmarks
    surface_raw: cdc.TrimeshSurface
    landmarks_raw: xr.DataArray

    # After normalize_axes + isolate_head (still before CTF alignment)
    surface_head: cdc.TrimeshSurface
    R_norm: np.ndarray  # normalize_axes rotation

    # CTF frame (this is what delete_masked_vertices operates on)
    surface_ctf: cdc.TrimeshSurface
    landmarks_ctf: xr.DataArray
    M_ctf: np.ndarray  # align_axes_from_landmarks affine

    # Mask + cap detection on surface_ctf
    cap_z: float
    cap_profile: dict[str, np.ndarray]  # keys: z, x_raw, x_smooth
    mask: np.ndarray
    mask_info: dict

    # Anonymized output
    surface_anon_ctf: cdc.TrimeshSurface
    surface_anon_dig: cdc.TrimeshSurface
    landmarks_dig: xr.DataArray


def run_pipeline(
    surface_raw: cdc.TrimeshSurface,
    landmarks_raw: xr.DataArray,
    subject: int = -1,
    landmark_keep_radius_mm: float = LANDMARK_KEEP_RADIUS_MM,
    ear_delete_radius_mm: float = EAR_DELETE_RADIUS_MM,
) -> PipelineArtifacts:
    """Run the full anonymization pipeline and return every intermediate.

    Mirrors the canonical `cedalion.geometry.photogrammetry.anonymization.
    anonymize_scan` orchestrator step-for-step, but exposes every
    intermediate (cap profile, mask, CTF-frame surfaces, both affines)
    that the validation notebooks need. Does not save to disk; saving
    is the caller's decision.

    Args:
        surface_raw: Raw Einstar scan (digitized frame).
        landmarks_raw: Five 10-20 landmarks in the same frame as surface_raw.
            Mixed-case labels (Lpa/Rpa) are normalized to canonical
            uppercase before use.
        subject: Optional subject number for bookkeeping only.
        landmark_keep_radius_mm: Per-landmark preservation sphere and
            half-width of the midline nasion strip.
        ear_delete_radius_mm: Sphere radius around LPA/RPA included in
            the deletion mask.

    Returns:
        PipelineArtifacts with every intermediate surface and the two
        affine transforms needed to invert the frame.
    """
    landmarks_raw = normalize_landmarks_labels(landmarks_raw)
    lm_raw = landmarks_raw.pint.dequantify().values
    idx = {lbl: i for i, lbl in enumerate(landmarks_raw["label"].values)}
    Nz_raw = lm_raw[idx["Nz"]]

    surface_n, _, R_norm = normalize_axes(surface_raw, Nz_raw)
    R_norm4 = np.eye(4)
    R_norm4[:3, :3] = R_norm
    crs_dim = next(d for d in landmarks_raw.dims if d != "label")
    landmarks_n = _transform_labeled_points(
        landmarks_raw,
        lambda p: _apply_affine(p, R_norm4),
        new_crs=crs_dim,
    )
    surface_head, _ = isolate_head(
        surface_n, landmarks_n.pint.dequantify().values[idx["Nz"]]
    )

    surface_ctf, landmarks_ctf, M_ctf = align_axes_from_landmarks(
        surface_head, landmarks_n
    )
    lm_ctf = landmarks_ctf.pint.dequantify().values
    idx_ctf = {lbl: i for i, lbl in enumerate(landmarks_ctf["label"].values)}

    verts = np.asarray(surface_ctf.mesh.vertices)
    Nz = lm_ctf[idx_ctf["Nz"]]
    Iz = lm_ctf[idx_ctf["Iz"]]
    Cz = lm_ctf[idx_ctf["Cz"]]
    Lpa = lm_ctf[idx_ctf["LPA"]]
    Rpa = lm_ctf[idx_ctf["RPA"]]

    cap_z, prof_z, prof_x_raw, prof_x_smooth = detect_cap_boundary(
        verts, Nz, Cz, Lpa, Rpa
    )
    ear_mid = 0.5 * (Lpa + Rpa)

    mask, mask_info = face_mask_from_landmarks(
        verts, Nz, Lpa, Rpa, cap_z=cap_z,
        ear_delete_radius=ear_delete_radius_mm,
    )

    for lm in (Nz, Iz, Cz, Lpa, Rpa):
        near = np.linalg.norm(verts - lm, axis=1) < landmark_keep_radius_mm
        mask[near] = False
    nasion_strip = (
        (verts[:, 2] >= Nz[2])
        & (verts[:, 2] < cap_z)
        & (np.abs(verts[:, 1] - Nz[1]) < landmark_keep_radius_mm)
        & (verts[:, 0] > ear_mid[0])
    )
    mask[nasion_strip] = False

    surface_anon_ctf = delete_masked_vertices(surface_ctf, mask)
    surface_anon_dig, landmarks_dig = revert_to_einstar_frame(
        surface_anon_ctf, landmarks_ctf, R_norm, M_ctf
    )

    return PipelineArtifacts(
        subject=subject,
        surface_raw=surface_raw,
        landmarks_raw=landmarks_raw,
        surface_head=surface_head,
        R_norm=R_norm,
        surface_ctf=surface_ctf,
        landmarks_ctf=landmarks_ctf,
        M_ctf=M_ctf,
        cap_z=float(cap_z),
        cap_profile={"z": prof_z, "x_raw": prof_x_raw, "x_smooth": prof_x_smooth},
        mask=mask,
        mask_info=mask_info,
        surface_anon_ctf=surface_anon_ctf,
        surface_anon_dig=surface_anon_dig,
        landmarks_dig=landmarks_dig,
    )
