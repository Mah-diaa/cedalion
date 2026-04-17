"""Face anonymization module for photogrammetry scans.

This module provides tools to anonymize facial regions in 3D photogrammetry
scans while preserving optode positions and anatomical landmarks for fNIRS
research. The anonymization complies with GDPR requirements while maintaining
scientific utility.

The pipeline works as follows:

1. User clicks the nasion (Nz) via ``pick_nasion``
2. Normalize axes so Y=anterior, Z=left
3. Isolate the head (remove body/shoulders)
4. Detect remaining landmarks (Iz, Cz, LPA, RPA) from nasion
5. Align the frame from the 5 landmarks and detect the cap boundary
6. Build the face + ear deletion mask from the landmarks
7. Delete the masked vertices

An alternative MediaPipe-based automatic pipeline lives on the
``auto-detection-pipeline`` branch.

Example:
    >>> from cedalion.geometry.photogrammetry.anonymization import (
    ...     pick_nasion, normalize_axes, isolate_head,
    ...     detect_landmarks_from_nasion, align_axes_from_landmarks,
    ...     detect_cap_boundary, face_mask_from_landmarks,
    ...     delete_masked_vertices,
    ... )
    >>> nasion = pick_nasion(surface)
    >>> surface, nasion, R = normalize_axes(surface, nasion)
    >>> surface, _ = isolate_head(surface, nasion)
    >>> landmarks = detect_landmarks_from_nasion(surface, nasion)

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from .face_detector import (
    normalize_axes,
    isolate_head,
    detect_landmarks_from_nasion,
    get_facial_region_mask_from_nasion,
    align_axes_from_landmarks,
    detect_cap_boundary,
    face_mask_from_landmarks,
    delete_masked_vertices,
)
from .ui import pick_nasion
from .validator import validate_anonymization


__all__ = [
    # Nasion detection
    "pick_nasion",
    # Axis normalization and head isolation
    "normalize_axes",
    "isolate_head",
    # Landmark and face mask detection
    "detect_landmarks_from_nasion",
    "get_facial_region_mask_from_nasion",
    # Landmark-only geometric pipeline
    "align_axes_from_landmarks",
    "detect_cap_boundary",
    "face_mask_from_landmarks",
    "delete_masked_vertices",
    # Validation
    "validate_anonymization",
]
