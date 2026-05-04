"""Rejected noise-perturbation operator (modification-based variant).

Used only by the three-way detectability comparison in notebooks 72 and
73. Numerically frozen under ``seed=NOISE_SEED``: lifted from commit
06d2282 (the ``anonymizer.py`` that produced the ``hero_color_noise_*``
figures) and kept here rather than in
``cedalion.geometry.photogrammetry.anonymization`` because the shipped
pipeline is delete-only.

Refactor only with the bit-exact identity check on Subject 17.
"""

from __future__ import annotations

import numpy as np

import trimesh
from scipy import sparse
from scipy.spatial import KDTree


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
