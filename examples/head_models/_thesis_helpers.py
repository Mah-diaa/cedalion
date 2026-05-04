"""Shared helpers for the thesis CSV-producing notebooks (64, 65, 66, 68-70, 72, 73).

Two kinds of code live here:

1. Pipeline orchestration and path/landmark conventions used by every
   CSV notebook (top of file).
2. The noise-perturbation operator (bottom of file), a rejected
   modification-based alternative used only by the three-way
   detectability comparison in notebooks 72 and 73. It is numerically
   frozen under `seed=42` so its CSV outputs stay reproducible across
   refactors; verify with the noise-identity check before any edit.

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
silently. Use this module from a notebook with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path().resolve()))
    from _thesis_helpers import (
        SUBJECTS, OPTODE_SUBJECTS, BARE_CAP_SUBJECTS,
        is_optode, s_id, load_raw, load_landmarks, run_pipeline,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

import cedalion
import cedalion.dataclasses as cdc
import cedalion.io
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


LANDMARK_KEEP_RADIUS_MM = 8.0
EAR_DELETE_RADIUS_MM = 40.0


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


# Noise-perturbation operator. Used only by notebooks 72 and 73 (the
# three-way detectability comparison and the S8 MediaPipe boxes notebook).
# Numerically frozen under `seed=NOISE_SEED`: lifted from commit 06d2282
# (the `anonymizer.py` that produced the `hero_color_noise_*` figures) and
# kept here rather than in `cedalion.geometry.photogrammetry.anonymization`
# because the shipped pipeline is delete-only. Refactor only with the
# bit-exact identity check on Subject 17.

NOISE_ITERATIONS = 80
NOISE_STRENGTH = 0.4
NOISE_LANDMARK_PROTECT_MM = 15.0
NOISE_BOUNDARY_TRANSITION_MM = 10.0
NOISE_BOUNDARY_LAMBDA = 0.3
NOISE_BOUNDARY_ITERATIONS = 10
NOISE_SEED = 42


def _deduplicate_vertices(
    vertices: np.ndarray, tol: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse coincident vertices and return the merged positions.

    Returns ``(merged_xyz, old_to_new)`` where ``old_to_new[i]`` is the
    merged-vertex index for original vertex ``i``. Used to dissolve
    Einstar UV-seam duplicates so the Laplacian sees a single connected
    surface.
    """
    n = len(vertices)
    tree = KDTree(vertices)
    pairs = tree.query_pairs(r=tol)
    if not pairs:
        return vertices.copy(), np.arange(n)

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        parent[i] = find(i)

    _, old_to_new = np.unique(parent, return_inverse=True)
    n_new = old_to_new.max() + 1

    sums = np.zeros((n_new, 3))
    counts = np.zeros(n_new)
    np.add.at(sums, old_to_new, vertices)
    np.add.at(counts, old_to_new, 1.0)
    return sums / counts[:, None], old_to_new


def _merge_seam_vertices(
    mesh: trimesh.Trimesh, tol: float = 0.01
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return a seam-merged trimesh plus the original-to-merged index map."""
    new_verts, old_to_new = _deduplicate_vertices(
        np.asarray(mesh.vertices), tol=tol
    )
    if len(new_verts) == len(mesh.vertices):
        return mesh.copy(), old_to_new

    new_faces = old_to_new[mesh.faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    merged_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces[valid], process=False
    )

    if hasattr(mesh.visual, "vertex_colors"):
        try:
            old_colors = np.asarray(
                mesh.visual.to_color().vertex_colors, dtype=np.float64
            )
            sums = np.zeros((len(new_verts), 4), dtype=np.float64)
            counts = np.zeros(len(new_verts), dtype=np.float64)
            np.add.at(sums, old_to_new, old_colors)
            np.add.at(counts, old_to_new, 1.0)
            merged_mesh.visual.vertex_colors = (
                sums / counts[:, None]
            ).astype(np.uint8)
        except Exception:
            pass

    return merged_mesh, old_to_new


def _unmerge_vertices(
    merged_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    old_to_new: np.ndarray,
) -> trimesh.Trimesh:
    """Re-expand a merged mesh back to the original vertex count and topology."""
    result_mesh = original_mesh.copy()
    result_mesh.vertices = merged_mesh.vertices[old_to_new]
    if hasattr(merged_mesh.visual, "vertex_colors"):
        try:
            colors = np.asarray(merged_mesh.visual.to_color().vertex_colors)
            result_mesh.visual.vertex_colors = colors[old_to_new]
        except Exception:
            pass
    result_mesh._cache.clear()
    return result_mesh


def _build_cotangent_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Cotangent-weighted Laplacian with ``L @ V = (weighted_neighbour_avg - V)``.

    Negative cotangents are clamped to zero (obtuse-angle edges contribute
    zero rather than disconnecting), plus a small 0.01-per-edge uniform
    bridge so every edge stays connected even when both adjacent angles
    are obtuse.
    """
    n = len(mesh.vertices)
    faces = mesh.faces
    verts = mesh.vertices
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]
    e01, e12, e20 = v1 - v0, v2 - v1, v0 - v2

    cross0 = np.maximum(np.linalg.norm(np.cross(e01, -e20), axis=1), 1e-8)
    cross1 = np.maximum(np.linalg.norm(np.cross(-e01, e12), axis=1), 1e-8)
    cross2 = np.maximum(np.linalg.norm(np.cross(-e12, e20), axis=1), 1e-8)
    cot0 = np.maximum(np.sum(e01 * (-e20), axis=1) / cross0, 0.0)
    cot1 = np.maximum(np.sum((-e01) * e12, axis=1) / cross1, 0.0)
    cot2 = np.maximum(np.sum((-e12) * e20, axis=1) / cross2, 0.0)

    rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    weights = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

    edges = mesh.edges_unique
    bridge_rows = np.concatenate([edges[:, 0], edges[:, 1]])
    bridge_cols = np.concatenate([edges[:, 1], edges[:, 0]])
    W = W + sparse.csr_matrix(
        (np.full(len(bridge_rows), 0.01), (bridge_rows, bridge_cols)),
        shape=(n, n),
    )

    degree = np.asarray(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return (sparse.diags(1.0 / degree) @ W - sparse.eye(n)).tocsr()


def _build_uniform_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Degree-normalised uniform Laplacian, used by the boundary smoothing pass."""
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    degree = np.asarray(A.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return (sparse.diags(1.0 / degree) @ A - sparse.eye(n)).tocsr()


def _compute_boundary_weights(
    mesh: trimesh.Trimesh, mask: np.ndarray, transition_width: float
) -> np.ndarray:
    """Cosine taper: 0 at the mask boundary, 1 inside, fading across ``transition_width``."""
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    e0, e1 = edges[:, 0], edges[:, 1]
    crossing = mask[e0] != mask[e1]
    boundary_vertices = np.unique(np.concatenate([
        e0[crossing & mask[e0]], e1[crossing & mask[e1]],
    ]))
    if not len(boundary_vertices):
        return mask.astype(float)

    masked_indices = np.where(mask)[0]
    weights = np.zeros(n)
    if not len(masked_indices):
        return weights

    distances, _ = KDTree(mesh.vertices[boundary_vertices]).query(
        mesh.vertices[masked_indices], k=1
    )
    inside = distances > transition_width
    weights[masked_indices[inside]] = 1.0
    near = ~inside
    t = distances[near] / transition_width
    weights[masked_indices[near]] = 0.5 * (1.0 - np.cos(np.pi * t))
    return weights


def _get_protected_indices(
    mesh: trimesh.Trimesh, landmarks_xyz: np.ndarray, radius: float
) -> np.ndarray:
    """Vertices within ``radius`` of any landmark; held still by the noise pass."""
    if not len(landmarks_xyz):
        return np.array([], dtype=int)
    d, _ = KDTree(landmarks_xyz).query(mesh.vertices, k=1)
    return np.where(d < radius)[0]


def _per_vertex_avg_edge_length(
    vertices: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Mean incident-edge length per vertex; one Gaussian sigma in stage 1."""
    el = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    n = len(vertices)
    sums = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(sums, edges[:, 0], el)
    np.add.at(sums, edges[:, 1], el)
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)
    counts[counts == 0] = 1
    return sums / counts


def noise_perturb_with_taper(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    landmarks_xyz: np.ndarray,
    *,
    iterations: int = NOISE_ITERATIONS,
    strength: float = NOISE_STRENGTH,
    protect_radius_mm: float = NOISE_LANDMARK_PROTECT_MM,
    boundary_transition_mm: float = NOISE_BOUNDARY_TRANSITION_MM,
    boundary_iterations: int = NOISE_BOUNDARY_ITERATIONS,
    boundary_lambda: float = NOISE_BOUNDARY_LAMBDA,
    seed: int = NOISE_SEED,
) -> trimesh.Trimesh:
    """Cotangent-Laplacian noise perturbation with a tapered boundary smoothing.

    Two stages, both bit-frozen for figure reproducibility:

    Stage 1 (noise): for each pass, every movable vertex (in ``mask``,
        outside ``protect_radius_mm`` of any landmark) is pulled
        ``strength`` of the way toward its cotangent-weighted neighbour
        mean and then perturbed by a Gaussian offset of one local edge
        length.

    Stage 2 (boundary): a uniform-Laplacian smoothing pass with a cosine
        taper that ramps the smoothing weight from 0 at the mask boundary
        to 1 inside, run for ``boundary_iterations`` iterations at
        ``boundary_lambda``. This blends the noisy facial region back
        into the surrounding scalp without a visible crack.

    Stage 0 collapses Einstar UV-seam duplicates first so the Laplacian
    sees a single connected mesh; the merge is undone at the end so the
    returned mesh has the original face topology and vertex count.
    """
    merged_mesh, old_to_new = _merge_seam_vertices(mesh)
    n_new = len(merged_mesh.vertices)

    merged_mask_int = np.zeros(n_new, dtype=int)
    np.add.at(merged_mask_int, old_to_new, mask.astype(int))
    merged_mask = merged_mask_int > 0

    protected = _get_protected_indices(merged_mesh, landmarks_xyz, protect_radius_mm)
    movable = merged_mask.copy()
    movable[protected] = False
    movable_idx = np.where(movable)[0]
    if not len(movable_idx):
        return _unmerge_vertices(merged_mesh, mesh, old_to_new)

    L_cot = _build_cotangent_laplacian(merged_mesh)
    V = merged_mesh.vertices.copy()
    sigma = _per_vertex_avg_edge_length(V, merged_mesh.edges_unique)

    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        V[movable_idx] += strength * (L_cot @ V)[movable_idx]
        V[movable_idx] += rng.standard_normal((len(movable_idx), 3)) * (
            sigma[movable_idx, None]
        )

    merged_mesh.vertices = V
    merged_mesh._cache.clear()

    weights = _compute_boundary_weights(
        merged_mesh, merged_mask, boundary_transition_mm
    )
    weights[protected] = 0.0
    if weights.sum() > 0:
        L_unif = _build_uniform_laplacian(merged_mesh)
        V = merged_mesh.vertices.copy()
        w = weights[:, None]
        for _ in range(boundary_iterations):
            V += boundary_lambda * w * (L_unif @ V)
        merged_mesh.vertices = V
        merged_mesh._cache.clear()

    return _unmerge_vertices(merged_mesh, mesh, old_to_new)
