"""Unit tests for face anonymization module."""

import numpy as np
import pytest
import trimesh
import xarray as xr
from numpy.testing import assert_allclose, assert_array_less

import cedalion
import cedalion.dataclasses as cdc
from cedalion import units

from cedalion.geometry.photogrammetry.anonymization import (
    detect_facial_landmarks,
    get_facial_region_mask,
    anonymize_facial_region,
    validate_anonymization,
    anonymize_scan,
    FacialLandmarks,
    FacialLandmarkType,
    AnonymizationConfig,
    AnonymizationMethod,
)
from cedalion.geometry.photogrammetry.anonymization.face_detector import (
    _normalize,
    _build_head_coordinate_system,
    _find_landmark,
)
from cedalion.geometry.photogrammetry.anonymization.anonymizer import (
    _build_adjacency_matrix,
    _compute_laplacian_displacement,
    smooth_region_selective,
)
from cedalion.geometry.photogrammetry.anonymization.validator import (
    validate_anonymization as validate_anonymization_fn,
    ValidationResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_sphere_surface():
    """Create a simple sphere mesh as a test surface."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=100)
    return cdc.TrimeshSurface(
        mesh=sphere, crs="scanner", units=cedalion.units.millimeter
    )


@pytest.fixture
def head_like_surface():
    """Create a more head-like elongated sphere surface."""
    # Create a sphere and scale it to be more head-shaped
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100)
    # Scale to be taller than wide (head-like proportions)
    vertices = sphere.vertices.copy()
    vertices[:, 2] *= 1.2  # Elongate vertically
    vertices[:, 1] *= 0.9  # Slightly narrow front-to-back

    head_mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
    return cdc.TrimeshSurface(
        mesh=head_mesh, crs="scanner", units=cedalion.units.millimeter
    )


@pytest.fixture
def anatomical_landmarks(simple_sphere_surface):
    """Create anatomical landmarks on a sphere surface."""
    # Place landmarks at anatomically reasonable positions on a sphere
    # Nz (nasion) - front
    nz = np.array([0, 100, 0])  # Front of sphere
    # Iz (inion) - back
    iz = np.array([0, -100, 0])  # Back of sphere
    # LPA (left preauricular) - left
    lpa = np.array([-100, 0, 0])  # Left of sphere
    # RPA (right preauricular) - right
    rpa = np.array([100, 0, 0])  # Right of sphere
    # Cz (vertex) - top
    cz = np.array([0, 0, 100])  # Top of sphere

    landmarks = np.array([nz, iz, lpa, rpa, cz])
    labels = ["Nz", "Iz", "LPA", "RPA", "Cz"]
    types = [cdc.PointType.LANDMARK] * 5

    return xr.DataArray(
        landmarks,
        dims=["label", "scanner"],
        coords={
            "label": labels,
            "type": ("label", types),
        },
        attrs={"units": "mm"},
    ).pint.quantify()


@pytest.fixture
def optode_positions(simple_sphere_surface):
    """Create some optode positions on the surface."""
    # Place optodes at various positions (avoiding the face)
    positions = np.array([
        [50, 0, 87],   # Upper right
        [-50, 0, 87],  # Upper left
        [30, -50, 80],  # Back right
        [-30, -50, 80],  # Back left
        [0, -70, 70],   # Back center
    ])
    labels = ["S1", "S2", "D1", "D2", "D3"]

    return xr.DataArray(
        positions,
        dims=["label", "scanner"],
        coords={
            "label": labels,
            "type": ("label", [cdc.PointType.SOURCE, cdc.PointType.SOURCE,
                              cdc.PointType.DETECTOR, cdc.PointType.DETECTOR,
                              cdc.PointType.DETECTOR]),
        },
        attrs={"units": "mm"},
    ).pint.quantify()


# ============================================================================
# Face Detector Tests
# ============================================================================


class TestFaceDetector:
    """Tests for face detection functionality."""

    def test_normalize(self):
        """Test vector normalization."""
        v = np.array([3, 4, 0])
        normalized = _normalize(v)
        assert_allclose(np.linalg.norm(normalized), 1.0)
        assert_allclose(normalized, [0.6, 0.8, 0.0])

    def test_normalize_zero_vector(self):
        """Test that zero vector remains unchanged."""
        v = np.array([0, 0, 0])
        normalized = _normalize(v)
        assert_allclose(normalized, [0, 0, 0])

    def test_build_head_coordinate_system(self, anatomical_landmarks):
        """Test head coordinate system construction."""
        landmarks = anatomical_landmarks.pint.dequantify()
        nz = landmarks.sel(label="Nz").values
        iz = landmarks.sel(label="Iz").values
        lpa = landmarks.sel(label="LPA").values
        rpa = landmarks.sel(label="RPA").values
        cz = landmarks.sel(label="Cz").values

        origin, x_axis, y_axis, z_axis = _build_head_coordinate_system(
            nz, iz, lpa, rpa, cz
        )

        # Origin should be midpoint of ears
        expected_origin = (lpa + rpa) / 2
        assert_allclose(origin, expected_origin)

        # All axes should be unit vectors
        assert_allclose(np.linalg.norm(x_axis), 1.0)
        assert_allclose(np.linalg.norm(y_axis), 1.0)
        assert_allclose(np.linalg.norm(z_axis), 1.0)

        # Axes should be orthogonal
        assert_allclose(np.dot(x_axis, y_axis), 0.0, atol=1e-10)
        assert_allclose(np.dot(x_axis, z_axis), 0.0, atol=1e-10)
        assert_allclose(np.dot(y_axis, z_axis), 0.0, atol=1e-10)

    def test_find_landmark_direct(self, anatomical_landmarks):
        """Test finding landmark by exact name."""
        nz = _find_landmark(anatomical_landmarks, "Nz")
        assert nz is not None
        assert len(nz) == 3

    def test_find_landmark_missing(self, anatomical_landmarks):
        """Test that missing landmark raises error."""
        with pytest.raises(ValueError):
            _find_landmark(anatomical_landmarks, "NonexistentLandmark")

    def test_detect_facial_landmarks(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test facial landmark detection."""
        result = detect_facial_landmarks(simple_sphere_surface, anatomical_landmarks)

        assert isinstance(result, FacialLandmarks)
        assert result.detection_method == "geometric"
        assert len(result.landmarks.label) == 6  # 6 facial landmarks

        # Check all expected landmarks are present
        expected = [
            FacialLandmarkType.LEFT_EYE.value,
            FacialLandmarkType.RIGHT_EYE.value,
            FacialLandmarkType.NOSE_TIP.value,
            FacialLandmarkType.NOSE_BRIDGE.value,
            FacialLandmarkType.MOUTH_CENTER.value,
            FacialLandmarkType.CHIN.value,
        ]
        for name in expected:
            assert name in result.landmarks.label.values

        # All confidences should be 1.0 for geometric detection
        for conf in result.confidence.values():
            assert conf == 1.0

    def test_get_facial_region_mask(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test facial region mask generation."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )

        mask = get_facial_region_mask(
            surface=simple_sphere_surface,
            facial_landmarks=facial_landmarks,
            protected_points=anatomical_landmarks,
            protection_radius=15.0 * units.mm,
        )

        # Mask should be boolean array with correct length
        assert mask.dtype == bool
        assert len(mask) == simple_sphere_surface.nvertices

        # Some vertices should be masked as facial region
        assert mask.sum() > 0
        # But not all vertices
        assert mask.sum() < len(mask)

    def test_mask_excludes_protected_points(
        self, simple_sphere_surface, anatomical_landmarks, optode_positions
    ):
        """Test that protection zones work correctly."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )

        # Combine landmarks and optodes
        protected = xr.concat(
            [anatomical_landmarks, optode_positions], dim="label"
        )

        mask = get_facial_region_mask(
            surface=simple_sphere_surface,
            facial_landmarks=facial_landmarks,
            protected_points=protected,
            protection_radius=15.0 * units.mm,
        )

        # Vertices near protected points should not be in mask
        vertices = simple_sphere_surface.mesh.vertices
        protected_positions = protected.pint.dequantify().values

        for pos in protected_positions:
            distances = np.linalg.norm(vertices - pos, axis=1)
            nearby_mask = distances < 15.0
            # No vertex within protection radius should be in facial mask
            assert not np.any(mask & nearby_mask)


# ============================================================================
# Anonymizer Tests
# ============================================================================


class TestAnonymizer:
    """Tests for anonymization functionality."""

    def test_build_adjacency_matrix(self, simple_sphere_surface):
        """Test adjacency matrix construction."""
        adjacency = _build_adjacency_matrix(simple_sphere_surface.mesh)

        # Each vertex should have neighbors
        for v in adjacency:
            assert len(adjacency[v]) > 0

        # Adjacency should be symmetric
        for v1, neighbors in adjacency.items():
            for v2 in neighbors:
                assert v1 in adjacency[v2]

    def test_compute_laplacian_displacement(self, simple_sphere_surface):
        """Test Laplacian displacement computation."""
        vertices = simple_sphere_surface.mesh.vertices
        adjacency = _build_adjacency_matrix(simple_sphere_surface.mesh)

        displacement = _compute_laplacian_displacement(vertices, adjacency)

        # Displacement should have same shape as vertices
        assert displacement.shape == vertices.shape

        # On a smooth surface like a sphere, displacements should be small
        # (vertices are already close to neighbor averages)
        max_disp = np.linalg.norm(displacement, axis=1).max()
        assert max_disp < 10.0  # Less than 10mm on a 100mm radius sphere

    def test_smooth_region_selective(self, simple_sphere_surface):
        """Test selective region smoothing."""
        # Create mask for half the sphere (by x coordinate)
        vertices = simple_sphere_surface.mesh.vertices
        mask = vertices[:, 0] > 0  # Right half

        # No protected indices
        protected = np.array([], dtype=int)

        smoothed = smooth_region_selective(
            mesh=simple_sphere_surface.mesh,
            mask=mask,
            protected_indices=protected,
            iterations=10,
            lamb=0.5,
            mu=-0.53,
        )

        # Smoothed mesh should have same number of vertices
        assert len(smoothed.vertices) == len(vertices)

        # Vertices in the masked region should have moved
        displacements = np.linalg.norm(smoothed.vertices - vertices, axis=1)
        assert displacements[mask].max() > 0

        # Vertices outside the mask should not have moved
        assert_allclose(displacements[~mask], 0, atol=1e-10)

    def test_smoothing_preserves_protected(self, simple_sphere_surface):
        """Test that protected vertices don't move during smoothing."""
        vertices = simple_sphere_surface.mesh.vertices

        # Full mask (smooth everything)
        mask = np.ones(len(vertices), dtype=bool)

        # Protect some specific vertices
        protected = np.array([0, 10, 50, 100])

        smoothed = smooth_region_selective(
            mesh=simple_sphere_surface.mesh,
            mask=mask,
            protected_indices=protected,
            iterations=50,
            lamb=0.5,
            mu=-0.53,
        )

        # Protected vertices should not have moved at all
        for idx in protected:
            assert_allclose(
                smoothed.vertices[idx], vertices[idx], atol=1e-10
            )

    def test_anonymize_facial_region(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test full facial region anonymization."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )
        mask = get_facial_region_mask(
            simple_sphere_surface, facial_landmarks, anatomical_landmarks
        )

        result = anonymize_facial_region(
            surface=simple_sphere_surface,
            facial_mask=mask,
            protected_points=anatomical_landmarks,
            config=AnonymizationConfig(smoothing_iterations=20),
        )

        # Result should contain all expected fields
        assert result.anonymized_surface is not None
        assert result.original_surface is simple_sphere_surface
        assert len(result.facial_mask) == simple_sphere_surface.nvertices
        assert len(result.vertex_displacements) == simple_sphere_surface.nvertices

        # Anonymized surface should have same topology
        assert result.anonymized_surface.nvertices == simple_sphere_surface.nvertices
        assert result.anonymized_surface.nfaces == simple_sphere_surface.nfaces

    def test_anonymization_methods(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test different anonymization methods."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )
        mask = get_facial_region_mask(
            simple_sphere_surface, facial_landmarks, anatomical_landmarks
        )

        for method in [AnonymizationMethod.SMOOTH, AnonymizationMethod.FLATTEN]:
            config = AnonymizationConfig(method=method, smoothing_iterations=10)
            result = anonymize_facial_region(
                simple_sphere_surface, mask, anatomical_landmarks, config
            )
            assert result.anonymized_surface is not None


# ============================================================================
# Validator Tests
# ============================================================================


class TestValidator:
    """Tests for validation sanity checks."""

    def test_validate_anonymization_passes(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test that validation passes for a correctly anonymized mesh."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )
        mask = get_facial_region_mask(
            simple_sphere_surface, facial_landmarks, anatomical_landmarks
        )

        result = anonymize_facial_region(
            simple_sphere_surface, mask, anatomical_landmarks,
            config=AnonymizationConfig(smoothing_iterations=5)
        )

        check = validate_anonymization(
            original_surface=simple_sphere_surface,
            anonymized_surface=result.anonymized_surface,
            facial_mask=mask,
            protected_points=anatomical_landmarks,
            tolerance=1.0 * units.mm,
        )

        assert isinstance(check, ValidationResult)
        assert check.mesh_valid
        assert check.protected_points_intact
        assert check.face_coverage_pct > 0

    def test_validate_returns_correct_counts(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test that vertex removal counts are consistent."""
        facial_landmarks = detect_facial_landmarks(
            simple_sphere_surface, anatomical_landmarks
        )
        mask = get_facial_region_mask(
            simple_sphere_surface, facial_landmarks, anatomical_landmarks
        )

        result = anonymize_facial_region(
            simple_sphere_surface, mask, anatomical_landmarks,
            config=AnonymizationConfig(smoothing_iterations=10)
        )

        check = validate_anonymization(
            simple_sphere_surface,
            result.anonymized_surface,
            mask,
            anatomical_landmarks,
        )

        assert check.expected_vertices_removed == int(mask.sum())
        assert check.actual_vertices_removed >= 0
        assert 0 <= check.face_coverage_pct <= 100
        assert check.protected_point_max_deviation >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(
        self, simple_sphere_surface, anatomical_landmarks, optode_positions
    ):
        """Test complete anonymize_scan workflow."""
        result = anonymize_scan(
            surface=simple_sphere_surface,
            anatomical_landmarks=anatomical_landmarks,
            optode_positions=optode_positions,
            config=AnonymizationConfig(smoothing_iterations=10),
            interactive=False,
            validate=True,
        )

        # Result should have anonymized surface
        assert result.anonymized_surface is not None

        # Topology should be preserved
        assert result.anonymized_surface.nvertices == simple_sphere_surface.nvertices
        assert result.anonymized_surface.nfaces == simple_sphere_surface.nfaces

    def test_pipeline_without_optodes(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test pipeline works without optode positions."""
        result = anonymize_scan(
            surface=simple_sphere_surface,
            anatomical_landmarks=anatomical_landmarks,
            optode_positions=None,
            interactive=False,
            validate=False,
        )

        assert result.anonymized_surface is not None

    def test_landmarks_preserved_within_tolerance(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test that landmarks are preserved within tolerance."""
        result = anonymize_scan(
            surface=simple_sphere_surface,
            anatomical_landmarks=anatomical_landmarks,
            protection_radius=20.0 * units.mm,
            interactive=False,
            validate=False,
        )

        # Check that validation passes with protected points
        check = validate_anonymization(
            simple_sphere_surface,
            result.anonymized_surface,
            result.facial_mask,
            anatomical_landmarks,
            tolerance=1.0 * units.mm,
        )

        assert check.protected_points_intact, (
            f"Protected points not intact: max deviation "
            f"{check.protected_point_max_deviation:.3f}mm"
        )

    def test_facial_region_modified(
        self, simple_sphere_surface, anatomical_landmarks
    ):
        """Test that facial region is actually modified."""
        result = anonymize_scan(
            surface=simple_sphere_surface,
            anatomical_landmarks=anatomical_landmarks,
            config=AnonymizationConfig(smoothing_iterations=50),
            interactive=False,
            validate=False,
        )

        # Facial region should have non-zero displacement
        facial_displacements = result.vertex_displacements[result.facial_mask]
        assert facial_displacements.mean() > 0, "Facial region should be modified"
