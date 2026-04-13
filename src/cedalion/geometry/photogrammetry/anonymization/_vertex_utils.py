"""Shared vertex utilities for the anonymization module.

Provides the core union-find algorithm for merging spatially coincident
vertices (seam duplicates in photogrammetry scans).

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import numpy as np
from scipy.spatial import KDTree


def deduplicate_vertices(
    vertices: np.ndarray, tol: float = 0.01
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Merge spatially coincident vertices using union-find.

    Groups vertices within ``tol`` mm of each other and averages their
    positions. Returns the merged vertices together with bidirectional
    index mappings so callers can remap faces, colors, or other per-vertex
    data.

    Args:
        vertices: Vertex positions of shape (N, 3).
        tol: Distance tolerance in mm for considering vertices coincident.

    Returns:
        Tuple of (merged_vertices, old_to_new, new_to_old) where:
        - merged_vertices: Averaged positions of shape (M, 3), M <= N.
        - old_to_new: Array of shape (N,) mapping old index to new index.
        - new_to_old: List of length M; new_to_old[j] lists old indices
          that were merged into new vertex j.
    """
    n = len(vertices)

    tree = KDTree(vertices)
    pairs = tree.query_pairs(r=tol)

    if not pairs:
        old_to_new = np.arange(n)
        new_to_old = [[i] for i in range(n)]
        return vertices.copy(), old_to_new, new_to_old

    # Union-find with path compression
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

    # Flatten parent pointers
    for i in range(n):
        parent[i] = find(i)

    # Build compact index mapping
    unique_roots, inverse = np.unique(parent, return_inverse=True)
    old_to_new = inverse
    n_new = len(unique_roots)

    # Build reverse mapping
    new_to_old = [[] for _ in range(n_new)]
    for old_idx in range(n):
        new_to_old[old_to_new[old_idx]].append(old_idx)

    # Average positions of merged vertices
    merged_vertices = np.zeros((n_new, 3))
    for new_idx in range(n_new):
        old_indices = new_to_old[new_idx]
        merged_vertices[new_idx] = vertices[old_indices].mean(axis=0)

    return merged_vertices, old_to_new, new_to_old
