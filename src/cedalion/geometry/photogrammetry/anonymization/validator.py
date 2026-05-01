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


logger = logging.getLogger("cedalion")


@dataclass
class ValidationResult:
    """Result of post-anonymization sanity check.

    Attributes:
        face_removed: True if vertex count dropped as expected
        mesh_valid: True if anonymized mesh has no degenerate faces
        expected_vertices_removed: Number of facial mask vertices
        actual_vertices_removed: Actual vertex count difference
        protected_points_preserved: True if every protected point's nearest
            vertex is bit-exact identical between original and anonymized mesh
        protected_point_max_delta_mm: max |d_anon - d_orig| across protected
            points; the deletion operator preserves surviving vertices
            bit-exact, so this must be 0 for a correct run
        face_coverage_pct: Percentage of original vertices in the facial mask
        passed: True if all checks passed
        summary: Human-readable one-line summary
    """

    face_removed: bool
    mesh_valid: bool
    expected_vertices_removed: int
    actual_vertices_removed: int
    protected_points_preserved: bool
    protected_point_max_delta_mm: float
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
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    protected_points: cdt.LabeledPoints,
) -> tuple[bool, float]:
    """Check that the deletion operator preserved every protected point.

    For each protected point, compute d_anon (distance to nearest vertex of
    the anonymized mesh) and d_orig (distance to nearest vertex of the
    original mesh) and report Delta = d_anon - d_orig. Since the deletion
    operator only drops vertices and never repositions a surviving one, the
    nearest vertex to a protected point is the same vertex in both meshes
    whenever it survives, and Delta must be bit-exact zero. Any nonzero
    Delta indicates a violation of the operator's invariant.

    Args:
        original_surface: Mesh before anonymization
        anonymized_surface: Mesh after anonymization
        protected_points: Points whose nearest-vertex identity must be
            preserved by the pipeline

    Returns:
        Tuple of (all_zero, max_abs_delta_mm)
    """
    if protected_points is None or len(protected_points.label) == 0:
        return True, 0.0

    positions = protected_points.pint.dequantify().values
    d_orig, _ = KDTree(original_surface.mesh.vertices).query(positions)
    d_anon, _ = KDTree(anonymized_surface.mesh.vertices).query(positions)
    delta = d_anon - d_orig

    max_abs_delta = float(np.max(np.abs(delta)))
    all_zero = max_abs_delta == 0.0

    if not all_zero:
        labels = [str(l) for l in protected_points.label.values]
        for label, d in zip(labels, delta):
            if d != 0.0:
                logger.warning(
                    f"Protected point '{label}' shifted by {d:+.3e} mm "
                    f"(d_anon - d_orig); operator invariant violated"
                )

    return all_zero, max_abs_delta


@cdc.validate_schemas
def validate_anonymization(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    facial_mask: np.ndarray,
    protected_points: cdt.LabeledPoints = None,
) -> ValidationResult:
    """Run post-anonymization sanity checks.

    Verifies that:
    1. The facial region was actually removed (vertex count dropped)
    2. The remaining mesh is valid (no degenerate faces)
    3. The deletion operator preserved every protected point's
       nearest-vertex identity (Delta = d_anon - d_orig must be 0)

    This is a practical bug-catcher, not a scientific validation metric.
    For scientific validation of fNIRS utility preservation, compare
    10-10 system positions on original vs anonymized surfaces.

    Args:
        original_surface: Original mesh before anonymization
        anonymized_surface: Mesh after anonymization
        facial_mask: Boolean mask of facial vertices on the original mesh
        protected_points: Landmarks and/or optodes whose nearest-vertex
            identity must be preserved by the pipeline (optional)

    Returns:
        ValidationResult with pass/fail and diagnostic info
    """
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

    # Check 3: Protected-point preservation (Delta = d_anon - d_orig == 0)
    points_preserved, max_delta = _check_protected_points(
        original_surface, anonymized_surface, protected_points
    )

    # Overall pass/fail
    passed = face_removed and mesh_valid and points_preserved

    # Build summary
    if passed:
        summary = (
            f"PASSED — {actual_removed:,} vertices removed "
            f"({face_coverage_pct:.1f}%), mesh valid, "
            f"protected points preserved (max |Delta| = 0)"
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
        if not points_preserved:
            issues.append(
                f"protected point shifted: max |Delta| = {max_delta:.3e} mm"
            )
        summary = f"FAILED — {'; '.join(issues)}"

    logger.info(f"Anonymization validation: {summary}")

    return ValidationResult(
        face_removed=face_removed,
        mesh_valid=mesh_valid,
        expected_vertices_removed=expected_removed,
        actual_vertices_removed=actual_removed,
        protected_points_preserved=points_preserved,
        protected_point_max_delta_mm=max_delta,
        face_coverage_pct=face_coverage_pct,
        passed=passed,
        summary=summary,
    )
