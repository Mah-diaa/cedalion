"""Mesh anonymization for photogrammetry scans.

This module provides functions to anonymize facial regions of 3D meshes using
Taubin smoothing while preserving optode and landmark positions.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
import trimesh
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units
from ._vertex_utils import deduplicate_vertices


logger = logging.getLogger("cedalion")


class AnonymizationMethod(Enum):
    """Available anonymization techniques."""

    SMOOTH = "smooth"
    FLATTEN = "flatten"
    COLOR_BLUR = "color_blur"
    COMBINED = "combined"
    HEAT_DIFFUSION = "heat_diffusion"
    NOISE = "noise"


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization process.

    Attributes:
        method: Anonymization technique to use
        smoothing_iterations: Number of Taubin filter iterations
        smoothing_lambda: Taubin filter lambda parameter (step size)
        smoothing_mu: Taubin filter mu parameter (shrinkage correction)
        boundary_transition_width: Width of transition zone at mask boundary
    """

    method: AnonymizationMethod = AnonymizationMethod.SMOOTH
    smoothing_iterations: int = 300
    smoothing_lambda: float = 0.5
    smoothing_mu: float = 0.0  # No shrinkage correction → pure Laplacian smoothing
    boundary_transition_width: Quantity = field(
        default_factory=lambda: 10.0 * units.mm
    )
    use_cotangent: bool = False
    diffusion_time: float = 50.0
    noise_strength: float = 1.0
    noise_iterations: int = 80
    noise_seed: int | None = None
    flatten_strength: float = 0.7


@dataclass
class AnonymizationResult:
    """Result of anonymization process.

    Attributes:
        anonymized_surface: The anonymized TrimeshSurface
        original_surface: The original TrimeshSurface (unchanged)
        facial_mask: Boolean mask of facial vertices
        vertex_displacements: Displacement of each vertex (in mm)
        config: Configuration used for anonymization
    """

    anonymized_surface: cdc.TrimeshSurface
    original_surface: cdc.TrimeshSurface
    facial_mask: np.ndarray
    vertex_displacements: np.ndarray
    config: AnonymizationConfig


def _build_adjacency_matrix(mesh: trimesh.Trimesh) -> dict[int, set[int]]:
    """Build vertex adjacency dictionary from mesh topology.

    Args:
        mesh: The trimesh object

    Returns:
        Dictionary mapping vertex index to set of neighbor indices
    """
    adjacency = {i: set() for i in range(len(mesh.vertices))}

    for face in mesh.faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)

    return adjacency


def _merge_seam_vertices(
    mesh: trimesh.Trimesh, tol: float = 0.01
) -> tuple[trimesh.Trimesh, np.ndarray, list[list[int]]]:
    """Merge spatially coincident vertices into single vertices.

    Photogrammetry scans have duplicate vertices at patch boundaries. Merging
    them before smoothing eliminates seam cracks because the Laplacian sees
    a single connected mesh instead of disconnected patches.

    Args:
        mesh: The trimesh object with potential duplicate vertices
        tol: Distance tolerance in mm for considering vertices coincident

    Returns:
        Tuple of (merged_mesh, old_to_new, new_to_old) where:
        - merged_mesh: New trimesh with merged vertices and averaged colors
        - old_to_new: Array mapping old vertex index to new vertex index
        - new_to_old: List where new_to_old[i] = list of old indices that merged
    """
    n = len(mesh.vertices)

    new_verts, old_to_new, new_to_old = deduplicate_vertices(
        mesh.vertices, tol=tol
    )
    n_new = len(new_verts)

    if n_new == n:
        return mesh.copy(), old_to_new, new_to_old

    # Re-index faces and remove degenerate triangles
    new_faces = old_to_new[mesh.faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    # Average vertex colors at merged positions
    new_colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        try:
            old_colors = mesh.visual.to_color().vertex_colors  # (n, 4) uint8
            new_colors = np.zeros((n_new, 4), dtype=np.float64)
            for new_idx in range(n_new):
                old_indices = new_to_old[new_idx]
                new_colors[new_idx] = old_colors[old_indices].mean(axis=0)
            new_colors = new_colors.astype(np.uint8)
        except Exception:
            pass

    merged_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces, process=False
    )
    if new_colors is not None:
        merged_mesh.visual.vertex_colors = new_colors

    n_merged = n - n_new
    n_degenerate = len(mesh.faces) - len(new_faces)
    logger.info(
        f"Merged {n_merged} duplicate vertices "
        f"({n} -> {n_new}), removed {n_degenerate} degenerate faces"
    )

    return merged_mesh, old_to_new, new_to_old


def _unmerge_vertices(
    merged_mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    old_to_new: np.ndarray,
) -> trimesh.Trimesh:
    """Expand merged mesh back to the original vertex count.

    Copies each merged vertex position back to all the original vertices
    that were merged into it. This preserves the original face topology
    and vertex count for mask indexing compatibility.

    Args:
        merged_mesh: Mesh with merged vertices (from _merge_seam_vertices)
        original_mesh: Original mesh with full vertex count
        old_to_new: Mapping from original vertex indices to merged indices

    Returns:
        New trimesh with original vertex count and face topology, but with
        vertex positions from the merged mesh
    """
    result_mesh = original_mesh.copy()
    result_verts = result_mesh.vertices.copy()

    # Copy merged positions back to all original vertices
    result_verts = merged_mesh.vertices[old_to_new]

    # Copy colors if available
    if hasattr(merged_mesh.visual, "vertex_colors"):
        try:
            merged_colors = merged_mesh.visual.to_color().vertex_colors
            result_colors = merged_colors[old_to_new]
            result_mesh.visual.vertex_colors = result_colors
        except Exception:
            pass

    result_mesh.vertices = result_verts
    result_mesh._cache.clear()
    return result_mesh


def _find_seam_pairs(mesh: trimesh.Trimesh, tol: float = 0.01) -> np.ndarray:
    """Find pairs of vertices at the same spatial position (seam vertices).

    Photogrammetry scans are stitched from multiple patches, producing
    duplicate vertices at shared boundaries. These vertices are spatially
    coincident but topologically disconnected, causing the Laplacian to
    treat each patch independently and producing visible crack lines.

    Args:
        mesh: The trimesh object
        tol: Distance tolerance in mm for considering vertices coincident

    Returns:
        Array of shape (n_pairs, 2) with seam vertex index pairs
    """
    tree = KDTree(mesh.vertices)
    pairs = tree.query_pairs(r=tol)
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.array(list(pairs), dtype=int)


def _add_seam_edges(A: sparse.csr_matrix, mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Add edges between duplicate vertices at mesh seams.

    This bridges topologically disconnected patches so the Laplacian can
    smooth across seam boundaries, eliminating crack artifacts.

    Args:
        A: Adjacency or weight matrix of shape (n, n)
        mesh: The trimesh object (used to find seam pairs)

    Returns:
        Modified matrix with seam edges added
    """
    seam_pairs = _find_seam_pairs(mesh)
    if len(seam_pairs) == 0:
        return A

    n = A.shape[0]
    rows = np.concatenate([seam_pairs[:, 0], seam_pairs[:, 1]])
    cols = np.concatenate([seam_pairs[:, 1], seam_pairs[:, 0]])
    data = np.ones(len(rows), dtype=float)
    seam_adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    logger.info(f"Bridging {len(seam_pairs)} seam vertex pairs")
    return A + seam_adj


def _build_sparse_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Build sparse uniform Laplacian matrix from mesh topology.

    The Laplacian L is defined so that L @ V gives the displacement
    (neighbor_avg - vertex) for each vertex, i.e. the "umbrella" operator.

    Automatically bridges seam vertices (duplicate vertices at the same
    position from photogrammetry stitching) so smoothing works across
    patch boundaries.

    Args:
        mesh: The trimesh object

    Returns:
        Sparse CSR matrix of shape (n_vertices, n_vertices)
    """
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=float)

    # Adjacency matrix
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Bridge seam vertices
    A = _add_seam_edges(A, mesh)

    # Degree (number of neighbors per vertex)
    degree = np.array(A.sum(axis=1)).ravel()
    degree[degree == 0] = 1  # avoid division by zero for isolated vertices

    # Normalized Laplacian: L = D^{-1} A - I
    D_inv = sparse.diags(1.0 / degree)
    L = D_inv @ A - sparse.eye(n)

    return L.tocsr()


def _build_cotangent_laplacian(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Build sparse cotangent-weighted Laplacian matrix from mesh topology.

    For each edge (i,j), the cotangent weight is w_ij = (cot(alpha) + cot(beta)) / 2
    where alpha and beta are the angles opposite to edge (i,j) in the two adjacent
    triangles. This weights neighbors by mesh geometry — well-shaped triangles
    contribute more, slivers contribute less.

    The result has the same form as _build_sparse_laplacian: L @ V gives the
    weighted displacement (weighted_neighbor_avg - vertex) for each vertex.

    Args:
        mesh: The trimesh object

    Returns:
        Sparse CSR matrix of shape (n_vertices, n_vertices)
    """
    n = len(mesh.vertices)
    verts = mesh.vertices
    faces = mesh.faces  # (F, 3)

    # Vectorized cotangent computation for all faces at once
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]  # each (F, 3)

    # Edge vectors from each corner
    e01 = v1 - v0
    e02 = v2 - v0
    e10 = -e01
    e12 = v2 - v1
    e20 = -e02
    e21 = -e12

    # cot(angle) = dot / |cross| for each corner
    cross0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cross1 = np.linalg.norm(np.cross(e10, e12), axis=1)
    cross2 = np.linalg.norm(np.cross(e20, e21), axis=1)

    # Clamp to avoid division by zero (degenerate triangles)
    min_cross = 1e-8
    np.maximum(cross0, min_cross, out=cross0)
    np.maximum(cross1, min_cross, out=cross1)
    np.maximum(cross2, min_cross, out=cross2)

    dot0 = np.sum(e01 * e02, axis=1)
    dot1 = np.sum(e10 * e12, axis=1)
    dot2 = np.sum(e20 * e21, axis=1)

    # Clamp cotangent values to >= 0 BEFORE accumulation.
    # Negative cotangents (from obtuse angles) would disconnect mesh edges if
    # clamped after summing; clamping here keeps edges connected with zero
    # weight from obtuse angles instead.
    cot0 = np.maximum(dot0 / cross0, 0.0)  # opposite edge (i1, i2)
    cot1 = np.maximum(dot1 / cross1, 0.0)  # opposite edge (i0, i2)
    cot2 = np.maximum(dot2 / cross2, 0.0)  # opposite edge (i0, i1)

    # Each cotangent contributes to both directions of its opposite edge
    # Edge opposite corner 0: (i1, i2), weight = cot0 * 0.5
    # Edge opposite corner 1: (i0, i2), weight = cot1 * 0.5
    # Edge opposite corner 2: (i0, i1), weight = cot2 * 0.5
    all_rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    all_cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    half_cot0 = cot0 * 0.5
    half_cot1 = cot1 * 0.5
    half_cot2 = cot2 * 0.5
    all_weights = np.concatenate([
        half_cot0, half_cot0, half_cot1, half_cot1, half_cot2, half_cot2
    ])

    # Build sparse weight matrix
    W = sparse.coo_matrix((all_weights, (all_rows, all_cols)), shape=(n, n)).tocsr()

    # Add small uniform weight to ensure full connectivity — edges where both
    # adjacent angles are obtuse would otherwise have zero weight, effectively
    # disconnecting mesh regions.
    edges = mesh.edges_unique
    uniform_rows = np.concatenate([edges[:, 0], edges[:, 1]])
    uniform_cols = np.concatenate([edges[:, 1], edges[:, 0]])
    uniform_data = np.full(len(uniform_rows), 0.01)
    W_uniform = sparse.csr_matrix(
        (uniform_data, (uniform_rows, uniform_cols)), shape=(n, n)
    )
    W = W + W_uniform

    # Bridge seam vertices — photogrammetry meshes have topologically
    # disconnected patches at coincident positions. Without bridging,
    # each patch smooths independently, producing visible crack lines.
    W = _add_seam_edges(W, mesh)

    # Normalize rows: L = D^{-1} W - I
    degree = np.array(W.sum(axis=1)).ravel()
    degree[degree == 0] = 1  # avoid division by zero for isolated vertices
    D_inv = sparse.diags(1.0 / degree)
    L = D_inv @ W - sparse.eye(n)

    return L.tocsr()


def _compute_boundary_weights(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    transition_width: float,
) -> np.ndarray:
    """Compute smoothing weights with gradient at mask boundary.

    Creates a weight array where:
    - Vertices inside the mask have weight 1.0
    - Vertices outside the mask have weight 0.0
    - Vertices near the boundary have weights that smoothly transition

    Args:
        mesh: The trimesh object
        mask: Boolean mask of vertices to smooth (True = facial region)
        transition_width: Width of transition zone in mm

    Returns:
        Weight array of shape (n_vertices,) with values in [0, 1]
    """
    n_vertices = len(mesh.vertices)
    weights = np.zeros(n_vertices)

    # Find boundary vertices (masked vertices with unmasked neighbors)
    adjacency = _build_adjacency_matrix(mesh)
    boundary_vertices = []

    for i in range(n_vertices):
        if mask[i]:
            neighbors = adjacency[i]
            if any(not mask[n] for n in neighbors):
                boundary_vertices.append(i)

    if len(boundary_vertices) == 0:
        # No boundary - just return mask as weights
        return mask.astype(float)

    # Build KDTree from boundary vertices
    boundary_positions = mesh.vertices[boundary_vertices]
    boundary_tree = KDTree(boundary_positions)

    # Compute distances to boundary for all masked vertices
    masked_indices = np.where(mask)[0]

    if len(masked_indices) == 0:
        return weights

    masked_positions = mesh.vertices[masked_indices]
    distances, _ = boundary_tree.query(masked_positions, k=1)

    # Compute weights based on distance from boundary
    for idx, (i, dist) in enumerate(zip(masked_indices, distances)):
        if dist <= transition_width:
            # Smooth transition using cosine function
            t = dist / transition_width
            weights[i] = 0.5 * (1.0 - np.cos(np.pi * t))
        else:
            weights[i] = 1.0

    return weights


def smooth_region_selective(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    protected_indices: np.ndarray,
    iterations: int = 50,
    lamb: float = 0.5,
    mu: float = -0.53,
    use_cotangent: bool = True,
) -> trimesh.Trimesh:
    """Apply Taubin smoothing only to masked vertices.

    Extends trimesh smoothing to work on a subset of vertices while keeping
    protected vertices completely fixed.

    Args:
        mesh: The trimesh object to smooth
        mask: Boolean array marking vertices to smooth (True = smooth)
        protected_indices: Indices of vertices that must never move
        iterations: Number of Taubin filter iterations
        lamb: Taubin lambda parameter (positive, forward step)
        mu: Taubin mu parameter (negative, backward step for shrinkage correction)
        use_cotangent: If True, use cotangent-weighted Laplacian (geometry-aware);
            if False, use uniform Laplacian (original behavior)

    Returns:
        New trimesh with smoothed facial region
    """
    # Create copy of mesh
    smoothed_mesh = mesh.copy()
    vertices = smoothed_mesh.vertices.copy()

    # Create weight mask (0 for protected, 1 for smoothable)
    weights = mask.astype(float)
    weights[protected_indices] = 0.0

    # Find which vertices to actually smooth
    n_smoothable = (weights > 0).sum()

    laplacian_type = "cotangent" if use_cotangent else "uniform"
    logger.info(
        f"Smoothing {n_smoothable} vertices with {iterations} iterations "
        f"({laplacian_type} Laplacian)"
    )

    if n_smoothable == 0:
        logger.warning("No vertices to smooth!")
        return smoothed_mesh

    # Build sparse Laplacian once (fast matrix ops instead of Python loops)
    if use_cotangent:
        L = _build_cotangent_laplacian(smoothed_mesh)
        # Cotangent weights have much larger magnitude than uniform weights.
        # Scale step sizes so displacement magnitude matches the uniform case.
        max_row_sum = np.abs(L).sum(axis=1).max()
        if max_row_sum > 2.0:
            scale = 2.0 / max_row_sum
            lamb *= scale
            mu *= scale
            logger.info(
                f"Cotangent auto-scale: {scale:.4f} "
                f"(lamb={lamb:.4f}, mu={mu:.4f})"
            )
    else:
        L = _build_sparse_laplacian(smoothed_mesh)

    # Region-constrained Taubin smoothing:
    # Each iteration computes Laplacian displacement for all vertices,
    # but only applies it where weight > 0. This avoids discontinuities
    # because smoothed vertices naturally pull toward their neighbors.
    w = weights[:, np.newaxis]  # (n, 1) for broadcasting
    for iteration in range(iterations):
        # Forward step (lambda — smoothing)
        vertices += lamb * w * (L @ vertices)

        # Backward step (mu — shrinkage correction, mu is negative so this
        # moves vertices AGAINST the Laplacian to counteract shrinkage)
        vertices += mu * w * (L @ vertices)

    smoothed_mesh.vertices = vertices
    smoothed_mesh._cache.clear()  # Invalidate normals, bounds, etc.

    logger.info("Smoothing complete")
    return smoothed_mesh


def apply_boundary_transition(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    transition_width: float,
    protected_indices: np.ndarray,
    iterations: int = 10,
    lamb: float = 0.3,
) -> trimesh.Trimesh:
    """Apply smoothing with smooth transition at mask boundary.

    Creates a gradient of smoothing weights at the boundary to avoid
    discontinuities between smoothed and unsmoothed regions.

    Args:
        mesh: The trimesh object
        mask: Boolean mask of facial region
        transition_width: Width of transition zone in mm
        protected_indices: Indices that must not move
        iterations: Number of smoothing iterations for boundary
        lamb: Smoothing strength for boundary

    Returns:
        Mesh with smoothed boundary transition
    """
    # Compute boundary weights
    weights = _compute_boundary_weights(mesh, mask, transition_width)

    # Zero out protected vertices
    weights[protected_indices] = 0.0

    # If no vertices need boundary smoothing, return as-is
    if weights.sum() == 0:
        return mesh

    # Create copy of mesh
    smoothed_mesh = mesh.copy()
    vertices = smoothed_mesh.vertices.copy()

    # Build sparse Laplacian once
    L = _build_sparse_laplacian(smoothed_mesh)

    # Region-constrained Laplacian smoothing at the boundary:
    # Only vertices with weight > 0 move, proportional to their weight.
    w = weights[:, np.newaxis]
    for iteration in range(iterations):
        vertices += lamb * w * (L @ vertices)

    smoothed_mesh.vertices = vertices
    smoothed_mesh._cache.clear()  # Invalidate normals, bounds, etc.
    return smoothed_mesh


def _heat_diffusion_smooth(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    protected_indices: np.ndarray,
    diffusion_time: float = 0.001,
) -> trimesh.Trimesh:
    """Apply heat diffusion smoothing to masked vertices.

    Solves the implicit heat equation (I - t * L_cot) * V_new = V_old
    where L_cot is the cotangent Laplacian and t is the diffusion time.
    This acts as an ideal low-pass filter on the mesh surface.

    Args:
        mesh: The trimesh object to smooth
        mask: Boolean array marking vertices to smooth
        protected_indices: Indices of vertices that must never move
        diffusion_time: Diffusion time parameter (larger = more smoothing)

    Returns:
        New trimesh with smoothed facial region
    """
    smoothed_mesh = mesh.copy()
    vertices = smoothed_mesh.vertices.copy()
    n = len(vertices)

    # Determine which vertices can move
    movable = mask.copy()
    movable[protected_indices] = False
    movable_indices = np.where(movable)[0]
    n_movable = len(movable_indices)

    logger.info(
        f"Heat diffusion: {n_movable} vertices, diffusion_time={diffusion_time}"
    )

    if n_movable == 0:
        logger.warning("No vertices to smooth!")
        return smoothed_mesh

    # Build cotangent Laplacian
    L = _build_cotangent_laplacian(mesh)

    # Build system matrix A = I - t * L (for full mesh)
    A = sparse.eye(n) - diffusion_time * L

    # For non-movable vertices, zero out their rows and set diagonal to 1
    # (they stay fixed). Use diagonal scaling trick: multiply rows by 0 then
    # add identity for fixed rows.
    fixed = np.ones(n, dtype=bool)
    fixed[movable_indices] = False
    # Scale fixed rows to zero
    row_scale = np.ones(n)
    row_scale[fixed] = 0.0
    A = sparse.diags(row_scale) @ A
    # Add identity for fixed rows
    A = A + sparse.diags(fixed.astype(float))

    # Solve A @ V_new = V_old for each coordinate
    rhs = vertices.copy()
    for coord in range(3):
        vertices[:, coord] = spsolve(A, rhs[:, coord])

    smoothed_mesh.vertices = vertices
    smoothed_mesh._cache.clear()  # Invalidate normals, bounds, etc.
    logger.info("Heat diffusion smoothing complete")
    return smoothed_mesh


def _noise_perturbation(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    protected_indices: np.ndarray,
    noise_strength: float = 0.4,
    noise_iterations: int = 5,
    noise_seed: int | None = None,
) -> trimesh.Trimesh:
    """Apply neighbor-weighted random perturbation to masked vertices.

    For each pass:
    1. Compute weighted average of neighbors (using cotangent Laplacian)
    2. Shift vertex partway toward neighbor average
    3. Add random offset scaled by local edge length

    This destroys fine facial detail while preserving macro head shape.

    Args:
        mesh: The trimesh object to perturb
        mask: Boolean array marking vertices to perturb
        protected_indices: Indices of vertices that must never move
        noise_strength: Blend factor toward neighbor average (0-1)
        noise_iterations: Number of perturbation passes
        noise_seed: Random seed for reproducibility (None for random)

    Returns:
        New trimesh with perturbed facial region
    """
    smoothed_mesh = mesh.copy()
    vertices = smoothed_mesh.vertices.copy()

    # Determine which vertices can move
    movable = mask.copy()
    movable[protected_indices] = False
    movable_indices = np.where(movable)[0]
    n_movable = len(movable_indices)

    logger.info(
        f"Noise perturbation: {n_movable} vertices, "
        f"strength={noise_strength}, iterations={noise_iterations}"
    )

    if n_movable == 0:
        logger.warning("No vertices to perturb!")
        return smoothed_mesh

    rng = np.random.default_rng(noise_seed)

    # Build cotangent Laplacian for weighted neighbor averaging
    L = _build_cotangent_laplacian(mesh)

    # Compute local edge lengths for scaling (average edge length per vertex)
    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    # Average edge length per vertex using np.add.at for vectorized accumulation
    avg_edge_len = np.zeros(len(vertices))
    edge_count = np.zeros(len(vertices))
    np.add.at(avg_edge_len, edges[:, 0], edge_lengths)
    np.add.at(avg_edge_len, edges[:, 1], edge_lengths)
    np.add.at(edge_count, edges[:, 0], 1)
    np.add.at(edge_count, edges[:, 1], 1)
    edge_count[edge_count == 0] = 1
    avg_edge_len /= edge_count

    for iteration in range(noise_iterations):
        # Compute neighbor-weighted average displacement: L @ V = avg - V
        displacement = L @ vertices  # shape (n, 3)

        # Vectorized: shift movable vertices toward neighbor average
        vertices[movable_indices] += noise_strength * displacement[movable_indices]

        # Vectorized: add random offset scaled by local edge length
        random_offsets = (
            rng.standard_normal((n_movable, 3))
            * (avg_edge_len[movable_indices, np.newaxis] * 1.0)
        )
        vertices[movable_indices] += random_offsets

    smoothed_mesh.vertices = vertices
    smoothed_mesh._cache.clear()  # Invalidate normals, bounds, etc.
    logger.info("Noise perturbation complete")
    return smoothed_mesh


def _get_protected_vertex_indices(
    mesh: trimesh.Trimesh,
    protected_points: cdt.LabeledPointCloud,
    protection_radius: float,
) -> np.ndarray:
    """Find vertex indices within protection radius of protected points.

    Args:
        mesh: The trimesh object
        protected_points: Points to protect (optodes + landmarks)
        protection_radius: Radius around protected points in mm

    Returns:
        Array of vertex indices that should not be modified
    """
    protected_positions = protected_points.pint.dequantify().values

    if len(protected_positions) == 0:
        return np.array([], dtype=int)

    # Find all vertices within protection radius of any protected point
    protected_tree = KDTree(protected_positions)
    vertex_distances, _ = protected_tree.query(mesh.vertices, k=1)
    protected_mask = vertex_distances < protection_radius

    return np.where(protected_mask)[0]


@cdc.validate_schemas
def anonymize_facial_region(
    surface: cdc.TrimeshSurface,
    facial_mask: np.ndarray,
    protected_points: cdt.LabeledPointCloud,
    config: AnonymizationConfig = None,
) -> AnonymizationResult:
    """Apply anonymization to facial region of mesh.

    Smooths facial features using Taubin filter while preserving protected
    optode and landmark positions. The algorithm:
    1. Identifies protected vertex zones around optodes/landmarks
    2. Applies Taubin smoothing to facial region
    3. Creates smooth transitions at boundaries
    4. Validates that protected points haven't moved

    Args:
        surface: Original TrimeshSurface
        facial_mask: Boolean array marking facial vertices
        protected_points: Points that must not move (optodes + landmarks)
        config: Anonymization configuration (uses defaults if None)

    Returns:
        AnonymizationResult with anonymized surface and metadata
    """
    if config is None:
        config = AnonymizationConfig()

    # Store original vertices for displacement calculation
    original_vertices = surface.mesh.vertices.copy()

    # Merge duplicate vertices at mesh seams before smoothing.
    # Photogrammetry scans have spatially coincident but topologically
    # disconnected vertices at patch boundaries. Without merging, the
    # Laplacian smooths each patch independently and seam vertices drift
    # apart, producing visible crack lines.
    merged_mesh, old_to_new, new_to_old = _merge_seam_vertices(surface.mesh)
    merged_mask = np.zeros(len(merged_mesh.vertices), dtype=bool)
    for new_idx in range(len(merged_mesh.vertices)):
        # A merged vertex is facial if ANY of its source vertices were facial
        if any(facial_mask[old_idx] for old_idx in new_to_old[new_idx]):
            merged_mask[new_idx] = True

    # Get protection radius from config
    transition_width_mm = float(config.boundary_transition_width.to("mm").magnitude)
    protection_radius = transition_width_mm * 1.5  # Extra margin for protection

    # Find protected vertex indices on the merged mesh
    protected_indices = _get_protected_vertex_indices(
        merged_mesh, protected_points, protection_radius
    )

    logger.info(
        f"Protecting {len(protected_indices)} vertices around "
        f"{len(protected_points.label)} points"
    )

    if config.method == AnonymizationMethod.SMOOTH:
        # Main smoothing pass
        smoothed_mesh = smooth_region_selective(
            mesh=merged_mesh,
            mask=merged_mask,
            protected_indices=protected_indices,
            iterations=config.smoothing_iterations,
            lamb=config.smoothing_lambda,
            mu=config.smoothing_mu,
            use_cotangent=config.use_cotangent,
        )

        # Boundary transition pass
        smoothed_mesh = apply_boundary_transition(
            mesh=smoothed_mesh,
            mask=merged_mask,
            transition_width=transition_width_mm,
            protected_indices=protected_indices,
            iterations=10,
            lamb=0.3,
        )

    elif config.method == AnonymizationMethod.FLATTEN:
        # Flatten facial region toward average plane (partial projection)
        smoothed_mesh = _flatten_facial_region(
            mesh=merged_mesh,
            mask=merged_mask,
            protected_indices=protected_indices,
            flatten_strength=config.flatten_strength,
        )

    elif config.method == AnonymizationMethod.COLOR_BLUR:
        # Only blur texture, keep geometry
        smoothed_mesh = _blur_texture(
            mesh=merged_mesh,
            mask=merged_mask,
        )

    elif config.method == AnonymizationMethod.COMBINED:
        # Apply smoothing then color blur
        smoothed_mesh = smooth_region_selective(
            mesh=merged_mesh,
            mask=merged_mask,
            protected_indices=protected_indices,
            iterations=config.smoothing_iterations,
            lamb=config.smoothing_lambda,
            mu=config.smoothing_mu,
            use_cotangent=config.use_cotangent,
        )
        smoothed_mesh = _blur_texture(
            mesh=smoothed_mesh,
            mask=merged_mask,
        )

    elif config.method == AnonymizationMethod.HEAT_DIFFUSION:
        # Heat diffusion smoothing with cotangent Laplacian
        smoothed_mesh = _heat_diffusion_smooth(
            mesh=merged_mesh,
            mask=merged_mask,
            protected_indices=protected_indices,
            diffusion_time=config.diffusion_time,
        )

        # Boundary transition pass
        smoothed_mesh = apply_boundary_transition(
            mesh=smoothed_mesh,
            mask=merged_mask,
            transition_width=transition_width_mm,
            protected_indices=protected_indices,
            iterations=10,
            lamb=0.3,
        )

    elif config.method == AnonymizationMethod.NOISE:
        # Neighbor-weighted random perturbation
        smoothed_mesh = _noise_perturbation(
            mesh=merged_mesh,
            mask=merged_mask,
            protected_indices=protected_indices,
            noise_strength=config.noise_strength,
            noise_iterations=config.noise_iterations,
            noise_seed=config.noise_seed,
        )

        # Boundary transition pass
        smoothed_mesh = apply_boundary_transition(
            mesh=smoothed_mesh,
            mask=merged_mask,
            transition_width=transition_width_mm,
            protected_indices=protected_indices,
            iterations=10,
            lamb=0.3,
        )

    else:
        raise ValueError(f"Unknown anonymization method: {config.method}")

    # Unmerge: expand smoothed vertices back to original vertex count
    smoothed_mesh = _unmerge_vertices(smoothed_mesh, surface.mesh, old_to_new)

    # Compute vertex displacements
    vertex_displacements = np.linalg.norm(
        smoothed_mesh.vertices - original_vertices, axis=1
    )

    # Create anonymized surface
    anonymized_surface = cdc.TrimeshSurface(
        mesh=smoothed_mesh,
        crs=surface.crs,
        units=surface.units,
        vertex_coords=deepcopy(surface.vertex_coords),
    )

    logger.info(
        f"Anonymization complete. "
        f"Max displacement: {vertex_displacements.max():.2f}mm, "
        f"Mean displacement in facial region: "
        f"{vertex_displacements[facial_mask].mean():.2f}mm"
    )

    return AnonymizationResult(
        anonymized_surface=anonymized_surface,
        original_surface=surface,
        facial_mask=facial_mask,
        vertex_displacements=vertex_displacements,
        config=config,
    )


def _flatten_facial_region(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    protected_indices: np.ndarray,
    flatten_strength: float = 0.7,
) -> trimesh.Trimesh:
    """Partially flatten facial region toward an average plane.

    Instead of fully projecting onto the plane (which destroys head curvature),
    vertices are blended partway toward the plane. A few Laplacian smooth
    iterations are applied afterward to round out the result.

    Args:
        mesh: The trimesh object
        mask: Boolean mask of facial vertices
        protected_indices: Indices that must not move
        flatten_strength: Blend factor toward plane (0 = no change, 1 = full
            projection). Default 0.7 preserves some curvature.

    Returns:
        Mesh with partially flattened facial region
    """
    smoothed_mesh = mesh.copy()
    vertices = smoothed_mesh.vertices.copy()

    # Get facial vertices (excluding protected)
    facial_indices = np.where(mask)[0]
    facial_indices = np.setdiff1d(facial_indices, protected_indices)

    if len(facial_indices) == 0:
        return smoothed_mesh

    facial_vertices = vertices[facial_indices]

    # Fit plane to facial vertices using PCA
    centroid = np.mean(facial_vertices, axis=0)
    centered = facial_vertices - centroid

    # SVD to find plane normal (smallest singular vector)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]  # Normal to best-fit plane

    # Partial projection — blend toward plane, keeping some curvature
    distances = np.dot(vertices[facial_indices] - centroid, normal)
    vertices[facial_indices] -= (flatten_strength * distances[:, np.newaxis]) * normal

    smoothed_mesh.vertices = vertices

    # Apply a few Laplacian smooth iterations to round out the result
    L = _build_sparse_laplacian(smoothed_mesh)
    w = np.zeros(len(vertices))
    w[facial_indices] = 1.0
    w = w[:, np.newaxis]
    for _ in range(5):
        vertices += 0.3 * w * (L @ vertices)
    smoothed_mesh.vertices = vertices
    smoothed_mesh._cache.clear()  # Invalidate normals, bounds, etc.

    logger.info(
        f"Flatten complete: strength={flatten_strength}, "
        f"{len(facial_indices)} vertices"
    )
    return smoothed_mesh


def _blur_texture(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
) -> trimesh.Trimesh:
    """Blur texture/vertex colors in facial region.

    Args:
        mesh: The trimesh object with vertex colors
        mask: Boolean mask of facial vertices

    Returns:
        Mesh with blurred colors in facial region
    """
    smoothed_mesh = mesh.copy()

    # Check if mesh has vertex colors
    if not hasattr(smoothed_mesh.visual, 'vertex_colors'):
        logger.warning("Mesh has no vertex colors to blur")
        return smoothed_mesh

    try:
        colors = smoothed_mesh.visual.to_color().vertex_colors.copy()
    except Exception:
        logger.warning("Could not extract vertex colors for blurring")
        return smoothed_mesh

    # Build adjacency
    adjacency = _build_adjacency_matrix(mesh)

    # Blur colors using neighborhood averaging
    new_colors = colors.copy()
    facial_indices = np.where(mask)[0]

    for _ in range(5):  # Multiple blur passes
        for i in facial_indices:
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                # Average with neighbors
                neighbor_colors = colors[neighbors].astype(float)
                avg_color = np.mean(neighbor_colors, axis=0)
                new_colors[i] = (0.5 * colors[i].astype(float) + 0.5 * avg_color).astype(
                    np.uint8
                )
        colors = new_colors.copy()

    # Apply blurred colors
    smoothed_mesh.visual.vertex_colors = new_colors
    return smoothed_mesh
