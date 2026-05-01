"""Shared helpers for the thesis Results chapter notebooks (51--59).

These notebooks evaluate the face-anonymization pipeline of
`cedalion.geometry.photogrammetry.anonymization` across eleven valid
thesis subjects:

- Optode-cap cohort (S1--S7): Subject 16, 17, 18, 19, 20, 21, 22 -- worn an
  fNIRS optode cap with cap-mounted optode markers at scan time.
- Bare-cap cohort (S8--S11): Subject 12, 13, 14, 15 -- worn a bare cap
  without optode markers. Included to demonstrate the pipeline generalises
  beyond the optode regime.

Subject 11 was acquired but is excluded because the scan is unusable
(acquisition-side defect). The optode co-registration test (notebook 55)
detects sticker markers on the cap, so it only applies to the optode cohort
and intentionally skips Subject 12--15. Every other measurement applies to
all eleven subjects.

These notebooks all need the same four things:

1. A path convention for the raw scan, the anonymized output, and the
   landmarks sidecar.
2. A way to load the five 10-20 landmarks for a given subject.
3. A one-call wrapper around the geometric pipeline that returns every
   intermediate artifact (isolated head, CTF landmarks, mask, anonymized
   mesh, digitized-frame anonymized mesh, ...) so that downstream
   notebooks do not need to re-implement the orchestration.
4. A SUBJECTS list and cohort predicates so every notebook iterates over
   the same cohort and emits a consistent `optode` / `s_id` column.

Landmarks come from a `{stem}_landmarks.tsv` sidecar written by
`save_anonymized_scan`. If that file does not exist for a given subject,
the helper raises a clear error rather than silently falling back.

This module is not part of cedalion itself; it lives next to the thesis
notebooks in examples/head_models/. To use it in a notebook:

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
from typing import Any

import numpy as np
import xarray as xr

import trimesh
from scipy import sparse
from scipy.spatial import KDTree

import cedalion
import cedalion.dataclasses as cdc
import cedalion.io
from cedalion.geometry.photogrammetry.anonymization import (
    align_axes_from_landmarks,
    delete_masked_vertices,
    detect_cap_boundary,
    face_mask_from_landmarks,
    isolate_head,
    normalize_axes,
    revert_to_einstar_frame,
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


def load_raw(n: int) -> cdc.TrimeshSurface:
    """Load the raw Einstar scan for Subject n (digitized frame)."""
    return cedalion.io.read_einstar_obj(str(subject_paths(n).obj))


def load_anon(n: int) -> cdc.TrimeshSurface:
    """Load the anonymized scan for Subject n (digitized frame)."""
    paths = subject_paths(n)
    if not paths.anon_exists:
        raise FileNotFoundError(
            f"No anonymized scan for Subject{n} at {paths.anon_obj}. "
            f"Run notebook 48 on this subject first."
        )
    return cedalion.io.read_einstar_obj(str(paths.anon_obj))


def load_landmarks(n: int) -> xr.DataArray:
    """Load the five 10-20 landmarks for Subject n (digitized frame).

    The landmarks sidecar is produced by `save_anonymized_scan(surface,
    path, landmarks=...)` as `{stem}_landmarks.tsv`. If it is missing,
    run notebook 48 on this subject first so the TSV is written.

    Returns:
        A LabeledPoints DataArray in the digitized frame, same convention
        as `read_einstar_obj` output.
    """
    paths = subject_paths(n)
    if not paths.landmarks_exist:
        raise FileNotFoundError(
            f"No landmarks TSV for Subject{n} at {paths.landmarks_tsv}. "
            f"Run notebook 48 on this subject first (it writes "
            f"`{paths.landmarks_tsv.name}` via save_anonymized_scan)."
        )
    return cedalion.io.load_tsv(str(paths.landmarks_tsv))


def available_subjects() -> list[int]:
    """Subjects whose raw scan and landmarks TSV both exist."""
    ready = []
    for n in SUBJECTS:
        paths = subject_paths(n)
        if paths.raw_exists and paths.landmarks_exist:
            ready.append(n)
    return ready


def missing_report() -> dict[int, list[str]]:
    """Map Subject n -> list of missing required files (raw, landmarks, anon)."""
    out: dict[int, list[str]] = {}
    for n in SUBJECTS:
        paths = subject_paths(n)
        missing: list[str] = []
        if not paths.raw_exists:
            missing.append("raw .obj")
        if not paths.landmarks_exist:
            missing.append("landmarks .tsv")
        if not paths.anon_exists:
            missing.append("anonymized .obj")
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
    mask_info: dict[str, Any]

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
) -> PipelineArtifacts:
    """Run the full anonymization pipeline and return every intermediate.

    This is the same sequence as notebook 48, collapsed into one call so
    the Results notebooks can share it. It does not save to disk; saving
    is the caller's decision.

    Args:
        surface_raw: Raw Einstar scan (digitized frame).
        landmarks_raw: Five 10-20 landmarks in the same frame as surface_raw.
        subject: Optional subject number for bookkeeping only.

    Returns:
        PipelineArtifacts with every intermediate surface and the two
        affine transforms needed to invert the frame.
    """
    lm_raw = landmarks_raw.pint.dequantify().values
    idx = {lbl: i for i, lbl in enumerate(landmarks_raw["label"].values)}
    Nz_raw = lm_raw[idx["Nz"]]

    surface_n, _, R_norm = normalize_axes(surface_raw, Nz_raw)
    lm_n_arr = lm_raw @ R_norm.T
    landmarks_n = (
        landmarks_raw.pint.dequantify().copy(data=lm_n_arr).pint.quantify()
    )
    surface_head, _ = isolate_head(surface_n, lm_n_arr[idx["Nz"]])

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

    ear_mid = 0.5 * (Lpa + Rpa)
    cap_z, prof_z, prof_x_raw, prof_x_smooth = detect_cap_boundary(
        verts, Nz, Cz, ear_mid, mid_y=0.5 * (Lpa[1] + Rpa[1])
    )

    mask, mask_info = face_mask_from_landmarks(
        verts, Nz=Nz, Iz=Iz, Cz=Cz, Lpa=Lpa, Rpa=Rpa, cap_z=cap_z,
        ear_delete_radius=EAR_DELETE_RADIUS_MM,
    )

    for lm in (Nz, Iz, Cz, Lpa, Rpa):
        near = np.linalg.norm(verts - lm, axis=1) < LANDMARK_KEEP_RADIUS_MM
        mask[near] = False
    nasion_strip = (
        (verts[:, 2] >= Nz[2])
        & (verts[:, 2] < cap_z)
        & (np.abs(verts[:, 1] - Nz[1]) < LANDMARK_KEEP_RADIUS_MM)
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


# Noise-perturbation operator (rejected modification-based alternative).
# Used by the three-way detectability comparison in notebook 59. Lives in this
# helper module rather than `cedalion.geometry.photogrammetry.anonymization`
# because the shipped pipeline is delete-only; this is a thesis-evaluation
# strawman, not a public API.
#
# Recovered verbatim from commit 06d2282 (`anonymizer.py`), which produced the
# hero noise PNGs in `Figures/results/hero_color_noise_*.png`. The pipeline is:
# (1) build a cotangent Laplacian (geometry-weighted), (2) for each pass shift
# masked vertices toward their neighbour mean and add per-vertex Gaussian
# offsets at one local-edge-length, (3) a separate boundary-transition pass
# that smoothly ramps the perturbation to zero across a 10 mm cosine band at
# the mask edge.
NOISE_ITERATIONS = 80
NOISE_STRENGTH = 0.4
NOISE_LANDMARK_PROTECT_MM = 15.0
NOISE_BOUNDARY_TRANSITION_MM = 10.0
NOISE_BOUNDARY_LAMBDA = 0.3
NOISE_BOUNDARY_ITERATIONS = 10
NOISE_SEED = 42


def _deduplicate_vertices(
    vertices: np.ndarray, tol: float = 0.01
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Merge spatially coincident vertices using union-find on a KDTree.

    Recovered verbatim from commit 06d2282
    (`anonymization/_vertex_utils.py::deduplicate_vertices`). Used to
    collapse Einstar UV-seam duplicates before the noise pass so the
    Laplacian sees one connected mesh and not thousands of tiny patches.
    """
    n = len(vertices)
    tree = KDTree(vertices)
    pairs = tree.query_pairs(r=tol)
    if not pairs:
        return vertices.copy(), np.arange(n), [[i] for i in range(n)]

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

    unique_roots, inverse = np.unique(parent, return_inverse=True)
    old_to_new = inverse
    n_new = len(unique_roots)

    new_to_old: list[list[int]] = [[] for _ in range(n_new)]
    for old_idx in range(n):
        new_to_old[old_to_new[old_idx]].append(old_idx)

    merged = np.zeros((n_new, 3))
    for new_idx in range(n_new):
        merged[new_idx] = vertices[new_to_old[new_idx]].mean(axis=0)
    return merged, old_to_new, new_to_old


def _merge_seam_vertices(
    mesh: trimesh.Trimesh, tol: float = 0.01
) -> tuple[trimesh.Trimesh, np.ndarray, list[list[int]]]:
    """Collapse coincident-position vertices into a single connected mesh.

    Recovered verbatim from commit 06d2282
    (`anonymizer.py::_merge_seam_vertices`). Returns the merged trimesh
    plus the index maps needed to expand back to the original vertex
    layout via `_unmerge_vertices`.
    """
    n = len(mesh.vertices)
    new_verts, old_to_new, new_to_old = _deduplicate_vertices(
        np.asarray(mesh.vertices), tol=tol
    )
    n_new = len(new_verts)

    if n_new == n:
        return mesh.copy(), old_to_new, new_to_old

    new_faces = old_to_new[mesh.faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    new_colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        try:
            old_colors = np.asarray(
                mesh.visual.to_color().vertex_colors, dtype=np.float64
            )
            new_colors = np.zeros((n_new, 4), dtype=np.float64)
            for new_idx in range(n_new):
                new_colors[new_idx] = old_colors[new_to_old[new_idx]].mean(axis=0)
            new_colors = new_colors.astype(np.uint8)
        except Exception:
            new_colors = None

    merged_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces, process=False
    )
    if new_colors is not None:
        merged_mesh.visual.vertex_colors = new_colors
    return merged_mesh, old_to_new, new_to_old


def _unmerge_vertices(
    merged_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    old_to_new: np.ndarray,
) -> trimesh.Trimesh:
    """Expand a merged mesh back to the original vertex count and topology.

    Recovered verbatim from commit 06d2282
    (`anonymizer.py::_unmerge_vertices`). Each original vertex receives
    the position of the merged vertex it was collapsed into, so the
    caller's mask, UV coordinates, and downstream consumers stay valid.
    """
    result_mesh = original_mesh.copy()
    result_mesh.vertices = merged_mesh.vertices[old_to_new]
    if hasattr(merged_mesh.visual, "vertex_colors"):
        try:
            merged_colors = np.asarray(merged_mesh.visual.to_color().vertex_colors)
            result_mesh.visual.vertex_colors = merged_colors[old_to_new]
        except Exception:
            pass
    result_mesh._cache.clear()
    return result_mesh


def _build_cotangent_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Cotangent-weighted Laplacian L with L @ V = (weighted_neighbour_avg - V).

    Vectorized cotangent computation across all faces, with negative
    cotangents clamped to zero (obtuse-angle edges contribute zero rather
    than disconnecting), plus a small uniform 0.01-per-edge bridge to keep
    every edge connected even when both adjacent angles are obtuse.
    """
    n = len(mesh.vertices)
    verts = mesh.vertices
    faces = mesh.faces

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]

    e01 = v1 - v0
    e02 = v2 - v0
    e10 = -e01
    e12 = v2 - v1
    e20 = -e02
    e21 = -e12

    cross0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cross1 = np.linalg.norm(np.cross(e10, e12), axis=1)
    cross2 = np.linalg.norm(np.cross(e20, e21), axis=1)
    np.maximum(cross0, 1e-8, out=cross0)
    np.maximum(cross1, 1e-8, out=cross1)
    np.maximum(cross2, 1e-8, out=cross2)

    cot0 = np.maximum(np.sum(e01 * e02, axis=1) / cross0, 0.0)
    cot1 = np.maximum(np.sum(e10 * e12, axis=1) / cross1, 0.0)
    cot2 = np.maximum(np.sum(e20 * e21, axis=1) / cross2, 0.0)

    all_rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    all_cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    half_cot0 = cot0 * 0.5
    half_cot1 = cot1 * 0.5
    half_cot2 = cot2 * 0.5
    all_weights = np.concatenate([
        half_cot0, half_cot0, half_cot1, half_cot1, half_cot2, half_cot2
    ])
    W = sparse.coo_matrix((all_weights, (all_rows, all_cols)), shape=(n, n)).tocsr()

    edges = mesh.edges_unique
    uniform_rows = np.concatenate([edges[:, 0], edges[:, 1]])
    uniform_cols = np.concatenate([edges[:, 1], edges[:, 0]])
    uniform_data = np.full(len(uniform_rows), 0.01)
    W_uniform = sparse.csr_matrix(
        (uniform_data, (uniform_rows, uniform_cols)), shape=(n, n)
    )
    W = W + W_uniform

    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    D_inv = sparse.diags(1.0 / degree)
    L = D_inv @ W - sparse.eye(n)
    return L.tocsr()


def _build_uniform_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Uniform-weight Laplacian (degree-normalised), used by the boundary pass."""
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=float)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    degree = np.array(A.sum(axis=1)).ravel()
    degree[degree == 0] = 1
    D_inv = sparse.diags(1.0 / degree)
    return (D_inv @ A - sparse.eye(n)).tocsr()


def _build_adjacency(mesh: trimesh.Trimesh) -> dict[int, set[int]]:
    adj = {i: set() for i in range(len(mesh.vertices))}
    for face in mesh.faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def _compute_boundary_weights(
    mesh: trimesh.Trimesh, mask: np.ndarray, transition_width: float
) -> np.ndarray:
    """Cosine taper from 0 at the mask boundary to 1 deep inside the mask."""
    n = len(mesh.vertices)
    weights = np.zeros(n)

    adjacency = _build_adjacency(mesh)
    boundary_vertices = [
        i for i in range(n)
        if mask[i] and any(not mask[nb] for nb in adjacency[i])
    ]
    if not boundary_vertices:
        return mask.astype(float)

    boundary_tree = KDTree(mesh.vertices[boundary_vertices])
    masked_indices = np.where(mask)[0]
    if not len(masked_indices):
        return weights

    distances, _ = boundary_tree.query(mesh.vertices[masked_indices], k=1)
    inside = distances > transition_width
    weights[masked_indices[inside]] = 1.0
    near = ~inside
    t = distances[near] / transition_width
    weights[masked_indices[near]] = 0.5 * (1.0 - np.cos(np.pi * t))
    return weights


def _get_protected_indices(
    mesh: trimesh.Trimesh, landmarks_xyz: np.ndarray, radius: float
) -> np.ndarray:
    if not len(landmarks_xyz):
        return np.array([], dtype=int)
    tree = KDTree(landmarks_xyz)
    d, _ = tree.query(mesh.vertices, k=1)
    return np.where(d < radius)[0]


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
    """Cotangent-Laplacian noise perturbation with a separate boundary smoothing pass.

    Verbatim port of the original `anonymizer.py::_noise_perturbation` plus
    `apply_boundary_transition` from commit 06d2282. The two-stage shape is the
    same as the operator that produced `hero_color_noise_*.png`:

    Stage 1 (noise): for each of `iterations` passes, every movable vertex
        (in `mask`, outside `protect_radius_mm` of any landmark) is pulled
        `strength` of the way toward its cotangent-weighted neighbour mean and
        then perturbed by a Gaussian offset of one local edge length.

    Stage 2 (boundary): a uniform-Laplacian smoothing pass with a cosine
        taper that ramps the smoothing weight from zero at the mask boundary
        to one inside, run for `boundary_iterations` iterations at
        `boundary_lambda`. This is what blends the noisy facial region back
        into the surrounding scalp without a visible crack.

    Args:
        mesh: Input trimesh (typically `surface_ctf.mesh`).
        mask: Boolean array, True for vertices in the facial region.
        landmarks_xyz: (k, 3) array of protected landmark positions in the
            same frame as `mesh.vertices`.

    Returns:
        A new trimesh with the masked region perturbed and the boundary
        smoothed; everything else is a bit-exact copy.
    """
    # Stage 0: collapse Einstar UV-seam duplicates so the Laplacian sees a
    # single connected mesh. Without this, every duplicate vertex drifts
    # independently per pass and visible cracks open along every patch
    # boundary. The merge is reverted at the end via _unmerge_vertices, so
    # the caller still receives a mesh with the original face topology and
    # vertex count.
    merged_mesh, old_to_new, new_to_old = _merge_seam_vertices(mesh)
    n_new = len(merged_mesh.vertices)

    # Lift the per-vertex facial mask onto the merged mesh: a merged vertex
    # is facial if any of its sources was.
    merged_mask = np.zeros(n_new, dtype=bool)
    for new_idx in range(n_new):
        if any(mask[old_idx] for old_idx in new_to_old[new_idx]):
            merged_mask[new_idx] = True

    protected = _get_protected_indices(
        merged_mesh, landmarks_xyz, protect_radius_mm
    )
    movable = merged_mask.copy()
    movable[protected] = False
    movable_idx = np.where(movable)[0]
    if not len(movable_idx):
        return _unmerge_vertices(merged_mesh, mesh, old_to_new)

    L_cot = _build_cotangent_laplacian(merged_mesh)
    edges = merged_mesh.edges_unique
    V = merged_mesh.vertices.copy()
    el = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    avg_edge = np.zeros(len(V))
    cnt = np.zeros(len(V))
    np.add.at(avg_edge, edges[:, 0], el)
    np.add.at(avg_edge, edges[:, 1], el)
    np.add.at(cnt, edges[:, 0], 1)
    np.add.at(cnt, edges[:, 1], 1)
    cnt[cnt == 0] = 1
    avg_edge /= cnt

    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        disp = L_cot @ V
        V[movable_idx] += strength * disp[movable_idx]
        V[movable_idx] += rng.standard_normal((len(movable_idx), 3)) * (
            avg_edge[movable_idx, None]
        )

    merged_mesh.vertices = V
    merged_mesh._cache.clear()

    # Stage 2: boundary-transition smoothing pass.
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

    # Stage 3: expand back to the original vertex count and topology.
    return _unmerge_vertices(merged_mesh, mesh, old_to_new)
