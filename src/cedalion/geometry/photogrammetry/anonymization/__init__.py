"""Face anonymization module for photogrammetry scans.

This module provides tools to anonymize facial regions in 3D photogrammetry
scans while preserving optode positions and anatomical landmarks for fNIRS
research. The anonymization complies with GDPR requirements while maintaining
scientific utility.

The pipeline works as follows:

1. User picks the 5 landmarks (Nz, Iz, Cz, LPA, RPA) via the upstream
   ``cedalion.vis.blocks.plot_surface(pick_landmarks=True)`` picker.
2. Normalize axes so Y=anterior, Z=left (``preprocessing``).
3. Isolate the head and remove disconnected fragments (``preprocessing``).
4. Align the full frame from the 5 landmarks (``preprocessing``).
5. Detect the cap boundary, build the deletion mask, and delete the masked
   vertices (``mask``).
6. Optionally revert the aligned surface and landmarks back to the raw
   Einstar frame via ``revert_to_einstar_frame`` so the saved files carry
   ``crs="digitized"`` and match the co-registration tutorial's input
   convention (``read_einstar_obj`` output).
7. Save the anonymized mesh (``.obj``) and the landmarks (``.tsv``) via
   ``save_anonymized_scan(surface, out_path, landmarks=...)``. The TSV is
   what the co-registration tutorial loads at step 5.2.

``detect_landmarks_from_nasion`` (``landmarks``) is an alternative that takes
just the nasion and geometrically infers the other four. An alternative
MediaPipe-based automatic pipeline lives on the ``auto-detection-pipeline``
branch.

Example:
    >>> from cedalion.geometry.photogrammetry.anonymization import (
    ...     normalize_axes, isolate_head, align_axes_from_landmarks,
    ...     detect_cap_boundary, face_mask_from_landmarks,
    ...     delete_masked_vertices,
    ... )
    >>> surface, nz, R = normalize_axes(surface, landmarks.sel(label='Nz'))
    >>> surface, _ = isolate_head(surface, nz)
    >>> surface, landmarks, R = align_axes_from_landmarks(surface, landmarks)

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from .preprocessing import (
    normalize_axes,
    isolate_head,
    align_axes_from_landmarks,
    revert_to_einstar_frame,
)
from .landmarks import detect_landmarks_from_nasion
from .mask import (
    detect_cap_boundary,
    face_mask_from_landmarks,
    delete_masked_vertices,
    save_anonymized_scan,
)


__all__ = [
    # Preprocessing (axis normalization, head isolation, full alignment,
    # and the inverse mapping back to the raw Einstar frame)
    "normalize_axes",
    "isolate_head",
    "align_axes_from_landmarks",
    "revert_to_einstar_frame",
    # Landmark detection
    "detect_landmarks_from_nasion",
    # Mask construction and application
    "detect_cap_boundary",
    "face_mask_from_landmarks",
    "delete_masked_vertices",
    "save_anonymized_scan",
]
