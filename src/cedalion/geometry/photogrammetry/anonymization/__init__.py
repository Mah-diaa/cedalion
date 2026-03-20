"""Face anonymization module for photogrammetry scans.

This module provides tools to anonymize facial regions in 3D photogrammetry
scans while preserving optode positions and anatomical landmarks for fNIRS
research. The anonymization complies with GDPR requirements while maintaining
scientific utility.

Example:
    >>> from cedalion.geometry.photogrammetry.anonymization import anonymize_scan
    >>> result = anonymize_scan(surface, landmarks, optodes)
    >>> anonymized = result.anonymized_surface

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units

from .face_detector import (
    detect_facial_landmarks,
    detect_landmarks_from_nasion,
    get_facial_region_mask,
    get_facial_region_center,
    estimate_face_bounding_box,
    FacialLandmarks,
    FacialLandmarkType,
    REQUIRED_LANDMARKS,
)
from .anonymizer import (
    anonymize_facial_region,
    smooth_region_selective,
    apply_boundary_transition,
    AnonymizationMethod,
    AnonymizationConfig,
    AnonymizationResult,
)
from .validator import (
    validate_anonymization,
    compute_point_deviations,
    compute_surface_distance,
    generate_validation_report,
    ValidationMetrics,
)
from .nasion_detector import detect_nasion_auto
from .ui import (
    FacialRegionEditor,
    AnonymizationPreview,
    DisplacementViewer,
    quick_preview,
    quick_displacement_view,
)


logger = logging.getLogger("cedalion")


def anonymize_scan(
    surface: cdc.TrimeshSurface,
    anatomical_landmarks: cdt.LabeledPointCloud = None,
    optode_positions: cdt.LabeledPointCloud = None,
    config: AnonymizationConfig = None,
    interactive: bool = False,
    validate: bool = True,
    protection_radius: Quantity = 15.0 * units.mm,
) -> AnonymizationResult:
    """Anonymize a photogrammetry scan while preserving critical points.

    Main entry point for the anonymization pipeline. This function:
    1. Detects the facial region using geometric heuristics from landmarks
    2. Creates protection zones around optodes and landmarks
    3. Applies Taubin smoothing to the facial region
    4. Optionally allows interactive refinement
    5. Validates that protected points haven't moved

    Args:
        surface: Textured TrimeshSurface from photogrammetry
        anatomical_landmarks: Known landmarks with at least Nz, Iz, LPA, RPA, Cz.
            Use detect_landmarks_from_nasion() to obtain these from a clicked Nz.
        optode_positions: Detected optode positions to protect (optional)
        config: Anonymization configuration (uses defaults if None)
        interactive: If True, allow user to refine facial region detection
        validate: If True, run validation and log metrics
        protection_radius: Radius around protected points that cannot be modified

    Returns:
        AnonymizationResult containing:
        - anonymized_surface: The anonymized TrimeshSurface
        - original_surface: The original surface (unchanged)
        - facial_mask: Boolean mask of facial vertices
        - vertex_displacements: How much each vertex moved
        - config: Configuration used

    Raises:
        ValueError: If anatomical_landmarks is not provided

    Example:
        >>> from cedalion.geometry.photogrammetry.anonymization import (
        ...     anonymize_scan, detect_landmarks_from_nasion,
        ... )
        >>>
        >>> # Detect landmarks from a user-clicked Nz point
        >>> landmarks = detect_landmarks_from_nasion(surface, nz_point)
        >>> result = anonymize_scan(surface, anatomical_landmarks=landmarks)
        >>>
        >>> # Access the anonymized mesh
        >>> anonymized_mesh = result.anonymized_surface
    """
    if config is None:
        config = AnonymizationConfig()

    logger.info("Starting face anonymization pipeline")
    logger.info(f"Surface has {surface.nvertices} vertices, {surface.nfaces} faces")

    # Landmarks must be provided
    if anatomical_landmarks is None:
        raise ValueError(
            "anatomical_landmarks must be provided. Use "
            "detect_landmarks_from_nasion(surface, nz_position) to detect "
            "landmarks from a user-clicked nasion point."
        )

    # Step 1: Detect facial landmarks
    logger.info("Detecting facial landmarks using geometric method")
    facial_landmarks = detect_facial_landmarks(surface, anatomical_landmarks)
    logger.info(
        f"Detected {len(facial_landmarks.landmarks.label)} facial landmarks: "
        f"{list(facial_landmarks.landmarks.label.values)}"
    )

    # Step 2: Combine all protected points
    protected_points = anatomical_landmarks
    if optode_positions is not None and len(optode_positions.label) > 0:
        protected_points = xr.concat(
            [anatomical_landmarks, optode_positions], dim="label"
        )
        logger.info(
            f"Protecting {len(anatomical_landmarks.label)} landmarks and "
            f"{len(optode_positions.label)} optodes"
        )
    else:
        logger.info(f"Protecting {len(anatomical_landmarks.label)} landmarks (no optodes)")

    # Step 3: Generate facial region mask
    logger.info("Generating facial region mask")
    facial_mask = get_facial_region_mask(
        surface=surface,
        facial_landmarks=facial_landmarks,
        protected_points=protected_points,
        protection_radius=protection_radius,
    )
    logger.info(
        f"Facial region: {facial_mask.sum()} vertices "
        f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
    )

    # Step 4: Interactive refinement (optional)
    if interactive:
        logger.info("Opening interactive facial region editor")
        editor = FacialRegionEditor(
            surface=surface,
            initial_mask=facial_mask,
            protected_points=protected_points,
            protection_radius=float(protection_radius.to("mm").magnitude),
        )
        facial_mask = editor.show()
        logger.info(
            f"Refined facial region: {facial_mask.sum()} vertices "
            f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
        )

    # Step 5: Apply anonymization
    logger.info(
        f"Applying {config.method.value} anonymization with "
        f"{config.smoothing_iterations} iterations"
    )
    result = anonymize_facial_region(
        surface=surface,
        facial_mask=facial_mask,
        protected_points=protected_points,
        config=config,
    )

    # Step 6: Validation
    if validate:
        logger.info("Validating anonymization quality")
        metrics = validate_anonymization(
            original_surface=surface,
            anonymized_surface=result.anonymized_surface,
            facial_mask=facial_mask,
            protected_points=protected_points,
            tolerance=0.5 * units.mm,
        )

        if not metrics.protected_points_preserved:
            logger.warning(
                "Some protected points moved beyond tolerance. "
                "Consider increasing protection_radius or reducing smoothing_iterations."
            )

        # Log summary
        report = generate_validation_report(metrics)
        for line in report.split("\n"):
            logger.debug(line)

    logger.info("Face anonymization complete")
    return result


__all__ = [
    # High-level API
    "anonymize_scan",
    # Auto nasion detection
    "detect_nasion_auto",
    # Face detection
    "detect_facial_landmarks",
    "detect_landmarks_from_nasion",
    "get_facial_region_mask",
    "get_facial_region_center",
    "estimate_face_bounding_box",
    "FacialLandmarks",
    "FacialLandmarkType",
    "REQUIRED_LANDMARKS",
    # Anonymization
    "anonymize_facial_region",
    "smooth_region_selective",
    "apply_boundary_transition",
    "AnonymizationMethod",
    "AnonymizationConfig",
    "AnonymizationResult",
    # Validation
    "validate_anonymization",
    "compute_point_deviations",
    "compute_surface_distance",
    "generate_validation_report",
    "ValidationMetrics",
    # UI
    "FacialRegionEditor",
    "AnonymizationPreview",
    "DisplacementViewer",
    "quick_preview",
    "quick_displacement_view",
]
