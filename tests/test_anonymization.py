"""Unit tests for face anonymization module."""

import numpy as np
import pytest
import trimesh
import xarray as xr
from numpy.testing import assert_allclose

import cedalion
import cedalion.dataclasses as cdc
from cedalion import units

from cedalion.geometry.photogrammetry.anonymization import (
    normalize_axes,
    isolate_head,
    detect_landmarks_from_nasion,
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
    """Create a more head-like elongated sphere surface.

    Axis convention: X=up, Y=anterior, Z=left.
    Nasion placed at (0, 100, 0) = front of sphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100)
    vertices = sphere.vertices.copy()
    vertices[:, 0] *= 1.2  # taller
    vertices[:, 1] *= 0.9  # slightly narrow front-to-back

    head_mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
    return cdc.TrimeshSurface(
        mesh=head_mesh, crs="scanner", units=cedalion.units.millimeter
    )


@pytest.fixture
def anatomical_landmarks():
    """Create anatomical landmarks on a sphere (already axis-normalized)."""
    landmarks = np.array([
        [0, 100, 0],    # Nz (front)
        [0, -100, 0],   # Iz (back)
        [120, 0, 0],    # Cz (top)
        [-100, 0, 0],   # LPA (left)  -- note: Z used for left in real coords
        [100, 0, 0],    # RPA (right)
    ], dtype=float)
    labels = ["Nz", "Iz", "Cz", "LPA", "RPA"]

    return xr.DataArray(
        landmarks,
        dims=["label", "scanner"],
        coords={
            "label": labels,
            "type": ("label", [cdc.PointType.LANDMARK] * 5),
        },
        attrs={"units": "mm"},
    ).pint.quantify()


# ============================================================================
# Axis Normalization
# ============================================================================


class TestNormalizeAxes:
    """Tests for axis normalization."""

    def test_rotation_matrix_orthogonal(self, simple_sphere_surface):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        nasion = np.array([0, 50, 50])
        _, _, R = normalize_axes(simple_sphere_surface, nasion)
        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_nasion_rotated_to_positive_y(self, simple_sphere_surface):
        """After normalization, nasion Y should be positive (anterior)."""
        nasion = np.array([0, 50, 50])  # forward in YZ diagonal
        _, rotated_nasion, _ = normalize_axes(simple_sphere_surface, nasion)
        # Y should now dominate (anterior direction)
        assert rotated_nasion[1] > 0

    def test_identity_when_already_aligned(self, simple_sphere_surface):
        """If nasion already on +Y axis, rotation should be ~identity."""
        nasion = np.array([0, 100, 0])
        _, rotated_nasion, R = normalize_axes(simple_sphere_surface, nasion)
        assert_allclose(R, np.eye(3), atol=1e-6)
        assert_allclose(rotated_nasion, nasion, atol=1e-6)


# ============================================================================
# Head Isolation
# ============================================================================


class TestIsolateHead:
    """Tests for head isolation."""

    def test_head_only_scan_unchanged(self, simple_sphere_surface):
        """A sphere (no body) should be returned largely unchanged."""
        nasion = np.array([0, 100, 0])
        head_surface, mask = isolate_head(simple_sphere_surface, nasion)
        # Sphere radius=100, default sphere radius=220 -> everything fits
        assert mask.mean() > 0.9

    def test_reduces_vertex_count_with_body(self):
        """Surface with body appended should have vertices removed."""
        # Head sphere
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=100)
        # Body box far below
        body = trimesh.creation.box(extents=[50, 50, 50])
        body.vertices[:, 0] -= 400  # move far below head
        combined = trimesh.util.concatenate([sphere, body])

        surface = cdc.TrimeshSurface(
            combined, crs="scanner", units=cedalion.units.millimeter
        )
        nasion = np.array([0, 100, 0])
        head_surface, mask = isolate_head(surface, nasion)
        assert head_surface.nvertices < surface.nvertices


# ============================================================================
# Landmark Detection from Nasion
# ============================================================================


class TestDetectLandmarksFromNasion:
    """Tests for landmark detection from nasion."""

    def test_returns_five_landmarks(self, head_like_surface):
        """Should detect exactly 5 landmarks."""
        nasion = np.array([0, 90, 0])  # front of elongated sphere
        landmarks = detect_landmarks_from_nasion(head_like_surface, nasion)

        assert len(landmarks.label) == 5
        expected_labels = {"Nz", "Iz", "Cz", "LPA", "RPA"}
        assert set(str(l) for l in landmarks.label.values) == expected_labels

    def test_cz_is_highest(self, head_like_surface):
        """Cz should have the highest X coordinate."""
        nasion = np.array([0, 90, 0])
        landmarks = detect_landmarks_from_nasion(head_like_surface, nasion)
        lm = landmarks.pint.dequantify()

        cz_x = float(lm.sel(label="Cz").values[0])
        for label in ["Nz", "Iz", "LPA", "RPA"]:
            other_x = float(lm.sel(label=label).values[0])
            assert cz_x >= other_x - 1.0, f"Cz should be highest, but {label} has higher X"

    def test_nz_position_preserved(self, head_like_surface):
        """Nz output should match the input nasion position."""
        nasion = np.array([0, 90, 0])
        landmarks = detect_landmarks_from_nasion(head_like_surface, nasion)
        nz_out = landmarks.sel(label="Nz").pint.dequantify().values
        assert_allclose(nz_out, nasion, atol=1e-6)


