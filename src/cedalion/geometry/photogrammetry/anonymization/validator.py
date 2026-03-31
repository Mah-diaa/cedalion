"""Post-anonymization sanity checks.

Validates that anonymization completed correctly: the expected facial region
was removed, the mesh is still valid, and protected points (optodes) remain
on the surface.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from dataclasses import dataclass
import logging

import numpy as np
from scipy.spatial import KDTree

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units


logger = logging.getLogger("cedalion")


@dataclass
class ValidationResult:
    """Result of post-anonymization sanity check.

    Attributes:
        face_removed: True if vertex count dropped as expected
        mesh_valid: True if anonymized mesh has no degenerate faces
        expected_vertices_removed: Number of facial mask vertices
        actual_vertices_removed: Actual vertex count difference
        protected_points_intact: True if all protected points are within
            tolerance of the anonymized surface
        protected_point_max_deviation: Worst-case deviation of any protected
            point from the anonymized surface (mm)
        face_coverage_pct: Percentage of original vertices in the facial mask
        passed: True if all checks passed
        summary: Human-readable one-line summary
    """

    face_removed: bool
    mesh_valid: bool
    expected_vertices_removed: int
    actual_vertices_removed: int
    protected_points_intact: bool
    protected_point_max_deviation: float
    face_coverage_pct: float
    passed: bool
    summary: str


def _check_mesh_valid(mesh) -> bool:
    """Check that a trimesh mesh has no degenerate faces.

    Args:
        mesh: A trimesh.Trimesh object

    Returns:
        True if mesh is valid (has faces, <1% degenerate)
    """
    if len(mesh.faces) == 0:
        return False

    # Check for degenerate faces (zero area)
    # Photogrammetry meshes commonly have a few thin triangles — only
    # fail if more than 1% of faces are degenerate.
    areas = mesh.area_faces
    n_degenerate = int(np.sum(areas < 1e-12))
    if n_degenerate > 0:
        pct = 100.0 * n_degenerate / len(mesh.faces)
        if pct > 1.0:
            logger.warning(
                f"Mesh has {n_degenerate} degenerate faces ({pct:.2f}%)"
            )
            return False
        else:
            logger.info(
                f"Mesh has {n_degenerate} degenerate faces ({pct:.4f}%) "
                f"— within tolerance for photogrammetry scans"
            )

    return True


def _check_protected_points(
    anonymized_surface: cdc.TrimeshSurface,
    protected_points: cdt.LabeledPointCloud,
    tolerance_mm: float,
) -> tuple[bool, float]:
    """Check that all protected points have a nearby vertex on the surface.

    Args:
        anonymized_surface: Mesh after anonymization
        protected_points: Points that should still be on the surface
        tolerance_mm: Maximum allowed distance from surface (mm)

    Returns:
        Tuple of (all_within_tolerance, max_deviation_mm)
    """
    if protected_points is None or len(protected_points.label) == 0:
        return True, 0.0

    positions = protected_points.pint.dequantify().values
    tree = KDTree(anonymized_surface.mesh.vertices)
    distances, _ = tree.query(positions)

    max_dev = float(np.max(distances))
    all_ok = max_dev <= tolerance_mm

    if not all_ok:
        labels = [str(l) for l in protected_points.label.values]
        for i, (label, dist) in enumerate(zip(labels, distances)):
            if dist > tolerance_mm:
                logger.warning(
                    f"Protected point '{label}' is {dist:.2f}mm from "
                    f"anonymized surface (tolerance: {tolerance_mm}mm)"
                )

    return all_ok, max_dev


@cdc.validate_schemas
def validate_anonymization(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    facial_mask: np.ndarray,
    protected_points: cdt.LabeledPointCloud = None,
    tolerance: Quantity = 1.0 * units.mm,
) -> ValidationResult:
    """Run post-anonymization sanity checks.

    Verifies that:
    1. The facial region was actually removed (vertex count dropped)
    2. The remaining mesh is valid (no degenerate faces)
    3. Protected points (optodes, landmarks) are still on the surface

    This is a practical bug-catcher, not a scientific validation metric.
    For scientific validation of fNIRS utility preservation, compare
    10-10 system positions on original vs anonymized surfaces.

    Args:
        original_surface: Original mesh before anonymization
        anonymized_surface: Mesh after anonymization
        facial_mask: Boolean mask of facial vertices on the original mesh
        protected_points: Landmarks and/or optodes that should still be
            reachable on the anonymized surface (optional)
        tolerance: Maximum allowed distance from protected points to the
            anonymized surface

    Returns:
        ValidationResult with pass/fail and diagnostic info
    """
    tolerance_mm = float(tolerance.to("mm").magnitude)

    # Check 1: Face removed — vertex count dropped
    orig_count = len(original_surface.mesh.vertices)
    anon_count = len(anonymized_surface.mesh.vertices)
    expected_removed = int(facial_mask.sum())
    actual_removed = orig_count - anon_count
    face_coverage_pct = 100.0 * expected_removed / orig_count if orig_count > 0 else 0.0

    # Allow some tolerance: boundary vertices may also be removed
    face_removed = actual_removed >= expected_removed * 0.9

    if not face_removed:
        logger.warning(
            f"Expected ~{expected_removed} vertices removed, "
            f"but only {actual_removed} were removed"
        )

    # Check 2: Mesh valid
    mesh_valid = _check_mesh_valid(anonymized_surface.mesh)

    # Check 3: Protected points still on surface
    points_intact, max_dev = _check_protected_points(
        anonymized_surface, protected_points, tolerance_mm
    )

    # Overall pass/fail
    passed = face_removed and mesh_valid and points_intact

    # Build summary
    if passed:
        summary = (
            f"PASSED — {actual_removed:,} vertices removed "
            f"({face_coverage_pct:.1f}%), mesh valid, "
            f"protected points within {tolerance_mm}mm"
        )
    else:
        issues = []
        if not face_removed:
            issues.append(
                f"vertex removal mismatch "
                f"(expected ~{expected_removed}, got {actual_removed})"
            )
        if not mesh_valid:
            issues.append("mesh has degenerate faces")
        if not points_intact:
            issues.append(
                f"protected point {max_dev:.2f}mm from surface "
                f"(tolerance: {tolerance_mm}mm)"
            )
        summary = f"FAILED — {'; '.join(issues)}"

    logger.info(f"Anonymization validation: {summary}")

    return ValidationResult(
        face_removed=face_removed,
        mesh_valid=mesh_valid,
        expected_vertices_removed=expected_removed,
        actual_vertices_removed=actual_removed,
        protected_points_intact=points_intact,
        protected_point_max_deviation=max_dev,
        face_coverage_pct=face_coverage_pct,
        passed=passed,
        summary=summary,
    )
