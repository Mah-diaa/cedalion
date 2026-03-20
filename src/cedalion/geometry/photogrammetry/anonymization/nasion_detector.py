"""Automatic nasion detection from 3D mesh geometry.

Detects the nasion (Nz) from an Einstar scan without user interaction.
Uses the known Einstar coordinate system directly:
    - X (index 0): Up
    - Y (index 1): Forward (toward face)
    - Z (index 2): Left of subject

Algorithm:
    1. Merge seam vertices (Einstar patch boundary duplicates)
    2. Find nose tip = max Y vertex in the top 40% of the head (by X)
    3. Extract midline anterior vertices above nose tip
    4. Bin by X height, take median Y per bin (robust to noise/seams)
    5. Nasion = first significant local minimum of Y above nose tip

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
from scipy.spatial import KDTree

import cedalion.dataclasses as cdc

logger = logging.getLogger("cedalion")


def detect_nasion_auto(
    surface: cdc.TrimeshSurface,
) -> tuple[np.ndarray, dict]:
    """Automatically detect the nasion from mesh geometry alone.

    Args:
        surface: TrimeshSurface from photogrammetry scan.

    Returns:
        Tuple of (nasion_position, metadata) where nasion_position is a
        numpy array of shape (3,) in mm, and metadata is a dict with
        detection details.
    """
    vertices = surface.mesh.vertices
    centroid = vertices.mean(axis=0)

    # --- Step 1: Merge seam vertices to eliminate crack noise ---
    merged_verts = _merge_close_vertices(vertices)

    # --- Step 2: Find nose tip ---
    nose_tip, _ = _find_nose_tip(merged_verts)

    # --- Step 3: Get midline anterior vertices above nose tip ---
    lat_dist = np.abs(merged_verts[:, 2] - nose_tip[2])
    midline = lat_dist < 10.0
    above = merged_verts[:, 0] > nose_tip[0]
    anterior = merged_verts[:, 1] > centroid[1]

    mask = midline & above & anterior
    candidates = np.where(mask)[0]

    if len(candidates) < 5:
        mask = midline & above
        candidates = np.where(mask)[0]

    if len(candidates) < 5:
        logger.warning("Auto nasion: too few candidates, returning nose tip")
        nasion = _snap_to_original(nose_tip, vertices)
        return nasion, {"method": "fallback", "confidence": 0.0,
                        "nose_tip": nose_tip.copy()}

    # --- Step 4: Bin by X height, take median Y per bin ---
    cand_x = merged_verts[candidates, 0]
    cand_y = merged_verts[candidates, 1]

    # Create bins of ~1mm height
    x_min, x_max = cand_x.min(), cand_x.max()
    n_bins = max(10, int((x_max - x_min) / 1.0))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.digitize(cand_x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Median Y per bin (robust to outliers/seam noise)
    bin_y = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = bin_indices == b
        if in_bin.sum() > 0:
            bin_y[b] = np.median(cand_y[in_bin])

    # Remove empty bins
    valid = ~np.isnan(bin_y)
    bin_centers_valid = bin_centers[valid]
    bin_y_valid = bin_y[valid]

    if len(bin_y_valid) < 3:
        logger.warning("Auto nasion: too few bins, returning nose tip")
        nasion = _snap_to_original(nose_tip, vertices)
        return nasion, {"method": "fallback", "confidence": 0.0,
                        "nose_tip": nose_tip.copy()}

    # --- Step 5: Find first significant local minimum ---
    # "Significant" = dip of at least 0.5mm compared to neighbors
    nasion_bin = _find_significant_minimum(bin_y_valid, min_dip=0.5)

    if nasion_bin is None:
        # Fallback: global minimum in first third
        third = max(1, len(bin_y_valid) // 3)
        nasion_bin = np.argmin(bin_y_valid[:third])

    nasion_x = bin_centers_valid[nasion_bin]
    nasion_y = bin_y_valid[nasion_bin]

    # Find the actual vertex closest to this binned position
    nasion_approx = np.array([nasion_x, nasion_y, nose_tip[2]])
    nasion = _snap_to_original(nasion_approx, vertices)

    # Confidence
    if nasion_bin > 0 and nasion_bin < len(bin_y_valid) - 1:
        dip = (min(bin_y_valid[nasion_bin - 1], bin_y_valid[nasion_bin + 1])
               - bin_y_valid[nasion_bin])
        confidence = float(min(1.0, dip / 5.0))
    else:
        confidence = 0.3

    metadata = {
        "method": "profile",
        "confidence": confidence,
        "nose_tip": nose_tip.copy(),
    }

    logger.info(
        f"Auto nasion: {nasion}, confidence={confidence:.2f}, "
        f"nose_tip={nose_tip}"
    )
    return nasion, metadata


def _merge_close_vertices(vertices: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Merge spatially coincident vertices (seam duplicates).

    Returns a deduplicated vertex array. Uses union-find to group vertices
    within tolerance, then averages their positions.

    Args:
        vertices: Mesh vertices of shape (N, 3).
        tol: Distance tolerance in mm.

    Returns:
        Unique vertices array of shape (M, 3) where M <= N.
    """
    tree = KDTree(vertices)
    pairs = tree.query_pairs(r=tol)

    if not pairs:
        return vertices.copy()

    n = len(vertices)
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
    n_new = len(unique_roots)

    new_verts = np.zeros((n_new, 3))
    counts = np.zeros(n_new)
    for i in range(n):
        new_verts[inverse[i]] += vertices[i]
        counts[inverse[i]] += 1
    new_verts /= counts[:, None]

    logger.info(f"Merged seam vertices: {n} -> {n_new}")
    return new_verts


def _find_nose_tip(vertices: np.ndarray) -> tuple[np.ndarray, int]:
    """Find nose tip as max Y vertex in the top 40% of the head.

    Args:
        vertices: Mesh vertices of shape (N, 3).

    Returns:
        Tuple of (nose_tip_position, vertex_index).
    """
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    x_range = x_max - x_min

    # Top 40% of the head by X (up)
    x_threshold = x_min + 0.6 * x_range
    upper_mask = vertices[:, 0] > x_threshold

    if upper_mask.sum() == 0:
        upper_mask = np.ones(len(vertices), dtype=bool)

    y_vals = vertices[:, 1].copy()
    y_vals[~upper_mask] = -np.inf
    idx = np.argmax(y_vals)

    return vertices[idx].copy(), idx


def _find_significant_minimum(values: np.ndarray, min_dip: float = 0.5):
    """Find the first local minimum that dips at least min_dip below neighbors.

    Args:
        values: 1D array of values.
        min_dip: Minimum dip depth to be considered significant.

    Returns:
        Index of the first significant local minimum, or None.
    """
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            dip = min(values[i - 1], values[i + 1]) - values[i]
            if dip >= min_dip:
                return i
    return None


def _snap_to_original(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Snap a point to the nearest original mesh vertex.

    Args:
        point: Target position of shape (3,).
        vertices: Original mesh vertices of shape (N, 3).

    Returns:
        Nearest vertex position as numpy array of shape (3,).
    """
    tree = KDTree(vertices)
    _, idx = tree.query(point)
    return vertices[idx].copy()
