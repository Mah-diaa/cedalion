"""Validation metrics for anonymization quality.

This module provides functions to validate that anonymization preserves
critical points (optodes, landmarks) while effectively modifying the facial
region.

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
class ValidationMetrics:
    """Comprehensive validation metrics for anonymization quality.

    Attributes:
        landmark_deviations: Euclidean deviation (mm) for each landmark
        optode_deviations: Euclidean deviation (mm) for each optode
        max_protected_deviation: Maximum deviation of any protected point (mm)
        mean_surface_distance: Mean surface-to-surface distance (mm)
        hausdorff_distance: Maximum surface-to-surface distance (mm)
        facial_displacement_mean: Mean vertex displacement in facial region (mm)
        facial_displacement_max: Maximum vertex displacement in facial region (mm)
        protected_points_preserved: True if all protected points within tolerance
    """

    landmark_deviations: dict[str, float]
    optode_deviations: dict[str, float]
    max_protected_deviation: float
    mean_surface_distance: float
    hausdorff_distance: float
    facial_displacement_mean: float
    facial_displacement_max: float
    protected_points_preserved: bool


def _find_nearest_surface_point(
    point: np.ndarray,
    mesh_vertices: np.ndarray,
    kdtree: KDTree = None,
) -> tuple[np.ndarray, float]:
    """Find the nearest point on a surface to a given point.

    Args:
        point: Query point coordinates
        mesh_vertices: Vertex positions of the mesh
        kdtree: Optional precomputed KDTree for faster queries

    Returns:
        Tuple of (nearest_point, distance)
    """
    if kdtree is None:
        kdtree = KDTree(mesh_vertices)

    distance, index = kdtree.query(point)
    nearest_point = mesh_vertices[index]

    return nearest_point, distance


def compute_point_deviations(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    points: cdt.LabeledPointCloud,
) -> dict[str, float]:
    """Compute surface deviation for each labeled point.

    For each point, finds the nearest vertex on both original and anonymized
    surfaces and computes the difference in position.

    Args:
        original_surface: Original mesh before anonymization
        anonymized_surface: Mesh after anonymization
        points: Labeled points to check (optodes, landmarks)

    Returns:
        Dictionary mapping point labels to deviation in mm
    """
    # Get point positions
    point_positions = points.pint.dequantify().values
    point_labels = [str(l) for l in points.label.values]

    # Build KDTrees for both surfaces
    original_tree = KDTree(original_surface.mesh.vertices)
    anonymized_tree = KDTree(anonymized_surface.mesh.vertices)

    deviations = {}

    for i, label in enumerate(point_labels):
        point = point_positions[i]

        # Find nearest vertex on original surface
        _, orig_idx = original_tree.query(point)
        orig_vertex = original_surface.mesh.vertices[orig_idx]

        # Find the same vertex index on anonymized surface
        anon_vertex = anonymized_surface.mesh.vertices[orig_idx]

        # Compute deviation
        deviation = np.linalg.norm(anon_vertex - orig_vertex)
        deviations[label] = float(deviation)

    return deviations


def compute_surface_distance(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    sample_points: int = 10000,
) -> tuple[float, float, float]:
    """Compute surface-to-surface distance metrics.

    Uses point-to-point distance between vertices to estimate surface
    distance. Samples vertices if there are too many.

    Args:
        original_surface: Original mesh
        anonymized_surface: Anonymized mesh
        sample_points: Maximum number of points to sample

    Returns:
        Tuple of (mean_distance, max_distance, hausdorff_distance) in mm
    """
    orig_vertices = original_surface.mesh.vertices
    anon_vertices = anonymized_surface.mesh.vertices

    # If meshes have same topology, compute direct vertex distances
    if len(orig_vertices) == len(anon_vertices):
        vertex_distances = np.linalg.norm(anon_vertices - orig_vertices, axis=1)
        mean_dist = float(np.mean(vertex_distances))
        max_dist = float(np.max(vertex_distances))
        hausdorff = max_dist

        return mean_dist, max_dist, hausdorff

    # Different topologies - use KDTree for nearest neighbor distances
    # Sample if too many vertices
    n_orig = len(orig_vertices)
    n_anon = len(anon_vertices)

    if n_orig > sample_points:
        indices = np.random.choice(n_orig, sample_points, replace=False)
        orig_sample = orig_vertices[indices]
    else:
        orig_sample = orig_vertices

    if n_anon > sample_points:
        indices = np.random.choice(n_anon, sample_points, replace=False)
        anon_sample = anon_vertices[indices]
    else:
        anon_sample = anon_vertices

    # Forward distances: original -> anonymized
    anon_tree = KDTree(anon_sample)
    forward_dists, _ = anon_tree.query(orig_sample)

    # Backward distances: anonymized -> original
    orig_tree = KDTree(orig_sample)
    backward_dists, _ = orig_tree.query(anon_sample)

    # Compute metrics
    mean_dist = float(np.mean(np.concatenate([forward_dists, backward_dists])))
    max_forward = float(np.max(forward_dists))
    max_backward = float(np.max(backward_dists))
    hausdorff = max(max_forward, max_backward)

    return mean_dist, hausdorff, hausdorff


def compute_facial_displacement_stats(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    facial_mask: np.ndarray,
) -> tuple[float, float]:
    """Compute displacement statistics for facial region.

    Args:
        original_surface: Original mesh
        anonymized_surface: Anonymized mesh
        facial_mask: Boolean mask of facial vertices

    Returns:
        Tuple of (mean_displacement, max_displacement) in mm
    """
    orig_vertices = original_surface.mesh.vertices
    anon_vertices = anonymized_surface.mesh.vertices

    # Ensure same topology
    if len(orig_vertices) != len(anon_vertices):
        logger.warning(
            "Surfaces have different vertex counts. Using nearest neighbor distances."
        )
        # Use KDTree approach
        facial_indices = np.where(facial_mask)[0]
        if len(facial_indices) == 0:
            return 0.0, 0.0

        orig_facial = orig_vertices[facial_indices]
        anon_tree = KDTree(anon_vertices)
        distances, _ = anon_tree.query(orig_facial)

        return float(np.mean(distances)), float(np.max(distances))

    # Direct computation for same topology
    displacements = np.linalg.norm(anon_vertices - orig_vertices, axis=1)
    facial_displacements = displacements[facial_mask]

    if len(facial_displacements) == 0:
        return 0.0, 0.0

    mean_disp = float(np.mean(facial_displacements))
    max_disp = float(np.max(facial_displacements))

    return mean_disp, max_disp


def _separate_landmarks_and_optodes(
    protected_points: cdt.LabeledPointCloud,
) -> tuple[cdt.LabeledPointCloud, cdt.LabeledPointCloud]:
    """Separate protected points into landmarks and optodes.

    Args:
        protected_points: All protected points

    Returns:
        Tuple of (landmarks, optodes) as LabeledPointClouds
    """
    # Common landmark names
    landmark_names = {
        "Nz", "Iz", "LPA", "RPA", "Cz",
        "Nas", "Nasion", "Inion", "Ini",
        "A1", "A2", "Lpa", "Rpa",
        "LeftPreauricular", "RightPreauricular",
        "Vertex", "CZ",
    }

    labels = [str(l) for l in protected_points.label.values]

    landmark_mask = np.array([l in landmark_names for l in labels])
    optode_mask = ~landmark_mask

    landmarks = protected_points.isel(label=np.where(landmark_mask)[0])
    optodes = protected_points.isel(label=np.where(optode_mask)[0])

    return landmarks, optodes


@cdc.validate_schemas
def validate_anonymization(
    original_surface: cdc.TrimeshSurface,
    anonymized_surface: cdc.TrimeshSurface,
    facial_mask: np.ndarray,
    protected_points: cdt.LabeledPointCloud,
    tolerance: Quantity = 0.5 * units.mm,
) -> ValidationMetrics:
    """Compute comprehensive validation metrics for anonymized scan.

    Validates that:
    1. Protected points (optodes, landmarks) haven't moved beyond tolerance
    2. Facial region has been effectively modified
    3. Overall surface integrity is maintained

    Args:
        original_surface: Original mesh before anonymization
        anonymized_surface: Mesh after anonymization
        facial_mask: Boolean mask of facial vertices
        protected_points: Landmarks + optodes that should be preserved
        tolerance: Maximum allowed deviation for protected points

    Returns:
        ValidationMetrics with comprehensive quality assessment
    """
    tolerance_mm = float(tolerance.to("mm").magnitude)

    # Separate landmarks and optodes
    landmarks, optodes = _separate_landmarks_and_optodes(protected_points)

    # Compute landmark deviations
    if len(landmarks.label) > 0:
        landmark_deviations = compute_point_deviations(
            original_surface, anonymized_surface, landmarks
        )
    else:
        landmark_deviations = {}

    # Compute optode deviations
    if len(optodes.label) > 0:
        optode_deviations = compute_point_deviations(
            original_surface, anonymized_surface, optodes
        )
    else:
        optode_deviations = {}

    # Maximum protected deviation
    all_deviations = list(landmark_deviations.values()) + list(optode_deviations.values())
    max_protected_deviation = max(all_deviations) if all_deviations else 0.0

    # Surface distance metrics
    mean_surface_distance, _, hausdorff_distance = compute_surface_distance(
        original_surface, anonymized_surface
    )

    # Facial displacement statistics
    facial_displacement_mean, facial_displacement_max = compute_facial_displacement_stats(
        original_surface, anonymized_surface, facial_mask
    )

    # Check if all protected points are within tolerance
    protected_points_preserved = max_protected_deviation <= tolerance_mm

    metrics = ValidationMetrics(
        landmark_deviations=landmark_deviations,
        optode_deviations=optode_deviations,
        max_protected_deviation=max_protected_deviation,
        mean_surface_distance=mean_surface_distance,
        hausdorff_distance=hausdorff_distance,
        facial_displacement_mean=facial_displacement_mean,
        facial_displacement_max=facial_displacement_max,
        protected_points_preserved=protected_points_preserved,
    )

    # Log summary
    logger.info(f"Validation results:")
    logger.info(f"  Max protected point deviation: {max_protected_deviation:.3f}mm")
    logger.info(f"  Protected points preserved: {protected_points_preserved}")
    logger.info(f"  Facial region mean displacement: {facial_displacement_mean:.2f}mm")
    logger.info(f"  Facial region max displacement: {facial_displacement_max:.2f}mm")

    if not protected_points_preserved:
        logger.warning(
            f"Protected points moved beyond tolerance ({tolerance_mm}mm). "
            f"Max deviation: {max_protected_deviation:.3f}mm"
        )

    return metrics


def generate_validation_report(
    metrics: ValidationMetrics,
    tolerance: Quantity = 0.5 * units.mm,
) -> str:
    """Generate a human-readable validation report.

    Args:
        metrics: ValidationMetrics from validate_anonymization
        tolerance: Tolerance threshold used

    Returns:
        Formatted string report
    """
    tolerance_mm = float(tolerance.to("mm").magnitude)

    lines = [
        "=" * 60,
        "ANONYMIZATION VALIDATION REPORT",
        "=" * 60,
        "",
        f"Protected Points Preserved: {'YES' if metrics.protected_points_preserved else 'NO'}",
        f"Tolerance: {tolerance_mm:.2f}mm",
        f"Maximum Protected Deviation: {metrics.max_protected_deviation:.3f}mm",
        "",
        "-" * 60,
        "LANDMARK DEVIATIONS (mm):",
        "-" * 60,
    ]

    for name, dev in sorted(metrics.landmark_deviations.items()):
        status = "OK" if dev <= tolerance_mm else "EXCEEDED"
        lines.append(f"  {name}: {dev:.3f} [{status}]")

    lines.extend([
        "",
        "-" * 60,
        "OPTODE DEVIATIONS (mm):",
        "-" * 60,
    ])

    for name, dev in sorted(metrics.optode_deviations.items()):
        status = "OK" if dev <= tolerance_mm else "EXCEEDED"
        lines.append(f"  {name}: {dev:.3f} [{status}]")

    lines.extend([
        "",
        "-" * 60,
        "SURFACE METRICS:",
        "-" * 60,
        f"  Mean Surface Distance: {metrics.mean_surface_distance:.3f}mm",
        f"  Hausdorff Distance: {metrics.hausdorff_distance:.3f}mm",
        "",
        "-" * 60,
        "FACIAL REGION MODIFICATION:",
        "-" * 60,
        f"  Mean Displacement: {metrics.facial_displacement_mean:.2f}mm",
        f"  Maximum Displacement: {metrics.facial_displacement_max:.2f}mm",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
