"""Private helpers shared by ``preprocessing`` and ``mask``.

Centralizes patterns that would otherwise drift across the pipeline:
texture-image lookup, face/vertex reindexing, affine application to plain
arrays and to ``LabeledPoints``, ear midpoint, and the upper-head centroid.
"""

from typing import Callable

import numpy as np

import cedalion.typing as cdt


def _resolve_texture_image(visual):
    """Return ``visual.image`` if set, else fall back to ``material.image``."""
    image = getattr(visual, "image", None)
    if image is not None:
        return image
    mat = getattr(visual, "material", None)
    return getattr(mat, "image", None) if mat is not None else None


def _reindex_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    kept_face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a triangle mesh by a face mask and rebuild dense vertex indices.

    Args:
        vertices: ``(N, 3)`` vertex array.
        faces: ``(F, 3)`` face index array.
        kept_face_mask: ``(F,)`` bool array.

    Returns:
        ``(new_vertices, new_faces, kept_vidx)`` where ``new_faces`` indexes
        into ``new_vertices`` and ``kept_vidx`` are the original vertex
        indices that survived (useful for slicing UVs / vertex colors).
    """
    kept_faces = faces[kept_face_mask]
    kept_vidx = np.unique(kept_faces)
    reindex = -np.ones(len(vertices), dtype=np.int64)
    reindex[kept_vidx] = np.arange(len(kept_vidx))
    new_faces = reindex[kept_faces]
    new_vertices = vertices[kept_vidx]
    return new_vertices, new_faces, kept_vidx


def _apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to an ``(N, 3)`` point array."""
    return points @ M[:3, :3].T + M[:3, 3]


def _transform_labeled_points(
    landmarks: cdt.LabeledPoints,
    transform_xyz: Callable[[np.ndarray], np.ndarray],
    new_crs: str,
) -> cdt.LabeledPoints:
    """Transform a LabeledPoints array and rename its spatial dim.

    Wraps the dequantify -> math -> ``copy(data=...)`` -> rename -> quantify
    cycle that ``align_axes_from_landmarks`` and ``revert_to_einstar_frame``
    both need.
    """
    dequant = landmarks.pint.dequantify()
    new_xyz = transform_xyz(dequant.values)
    old_crs_dim = next(d for d in landmarks.dims if d != "label")
    return (
        dequant.copy(data=new_xyz)
        .rename({old_crs_dim: new_crs})
        .pint.quantify()
    )


def _ear_midpoint(Lpa: np.ndarray, Rpa: np.ndarray) -> np.ndarray:
    """Return the LPA/RPA midpoint (CTF origin in the aligned frame)."""
    return 0.5 * (Lpa + Rpa)


def _upper_head_centroid(
    vertices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Centroid of vertices in the top 40% of the X range.

    Returns ``(centroid_xyz, x_max)``. Callers that only need the YZ
    components zero out X themselves.
    """
    x_max = float(vertices[:, 0].max())
    x_min = float(vertices[:, 0].min())
    upper = vertices[vertices[:, 0] > x_min + 0.6 * (x_max - x_min)]
    centroid = upper.mean(axis=0)
    return centroid, x_max
