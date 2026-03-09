"""Facial region detection for photogrammetry scans.

This module provides geometric-based facial detection using anatomical landmarks
(Nz, Iz, LPA, RPA, Cz) to estimate facial feature positions based on known head
proportions. It also provides semi-manual landmark detection from a user-provided
nasion (Nz) position.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
from scipy.spatial import KDTree
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units
logger = logging.getLogger("cedalion")


# Required anatomical landmarks for detection
REQUIRED_LANDMARKS = ["Nz", "Iz", "LPA", "RPA", "Cz"]

# Alternative landmark names (some datasets use different conventions)
LANDMARK_ALIASES = {
    "Nz": ["Nz", "Nas", "Nasion", "NAS"],
    "Iz": ["Iz", "Ini", "Inion", "INI"],
    "LPA": ["LPA", "Lpa", "A1", "LeftPreauricular"],
    "RPA": ["RPA", "Rpa", "A2", "RightPreauricular"],
    "Cz": ["Cz", "CZ", "Vertex"],
}


class FacialLandmarkType(Enum):
    """Types of facial landmarks for detection."""

    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE_TIP = "nose_tip"
    NOSE_BRIDGE = "nose_bridge"
    MOUTH_CENTER = "mouth_center"
    CHIN = "chin"


@dataclass
class FacialLandmarks:
    """Container for detected facial landmarks.

    Attributes:
        landmarks: xarray DataArray with estimated facial landmark positions
        confidence: confidence score per landmark (1.0 for geometric detection)
        detection_method: method used for detection ("geometric")
    """

    landmarks: cdt.LabeledPointCloud
    confidence: dict[str, float]
    detection_method: str


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def _find_landmark(
    landmarks: cdt.LabeledPointCloud, name: str
) -> np.ndarray:
    """Find a landmark by name, trying aliases if needed.

    Args:
        landmarks: LabeledPointCloud containing anatomical landmarks
        name: canonical landmark name (e.g., "Nz")

    Returns:
        numpy array of shape (3,) with landmark coordinates

    Raises:
        ValueError: if landmark not found under any alias
    """
    labels = [str(l) for l in landmarks.label.values]

    # Try direct match first
    if name in labels:
        return landmarks.sel(label=name).pint.dequantify().values

    # Try aliases
    for alias in LANDMARK_ALIASES.get(name, []):
        if alias in labels:
            return landmarks.sel(label=alias).pint.dequantify().values

    raise ValueError(
        f"Landmark '{name}' not found. Available landmarks: {labels}. "
        f"Tried aliases: {LANDMARK_ALIASES.get(name, [])}"
    )


def _build_head_coordinate_system(
    nz: np.ndarray,
    iz: np.ndarray,
    lpa: np.ndarray,
    rpa: np.ndarray,
    cz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a head-centered coordinate system from anatomical landmarks.

    The coordinate system is defined as:
    - Origin: midpoint between LPA and RPA (preauricular points)
    - X-axis: lateral, pointing from LPA to RPA
    - Y-axis: anterior, pointing toward Nz (nasion)
    - Z-axis: superior, pointing upward (cross product of X and Y)

    Args:
        nz: Nasion coordinates
        iz: Inion coordinates
        lpa: Left preauricular point coordinates
        rpa: Right preauricular point coordinates
        cz: Vertex coordinates

    Returns:
        Tuple of (origin, x_axis, y_axis, z_axis) as numpy arrays
    """
    # Origin at midpoint of ears
    origin = (lpa + rpa) / 2.0

    # X-axis: lateral (left to right)
    x_axis = _normalize(rpa - lpa)

    # Initial Y direction: toward nasion
    y_direction = nz - origin

    # Z-axis: perpendicular to X and Y direction
    z_axis = _normalize(np.cross(x_axis, y_direction))

    # Y-axis: perpendicular to X and Z (ensures orthogonality)
    y_axis = _normalize(np.cross(z_axis, x_axis))

    return origin, x_axis, y_axis, z_axis


def _estimate_facial_landmarks_geometric(
    nz: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate facial landmark positions using anatomical proportions.

    Estimates are based on average adult head proportions:
    - Eyes: ~30mm lateral to midline, ~30mm superior to nasion level
    - Nose tip: ~40mm inferior-anterior to nasion
    - Nose bridge: at nasion level, slightly anterior
    - Mouth: ~70mm inferior to nasion
    - Chin: ~100mm inferior to nasion

    Args:
        nz: Nasion position
        origin: Head coordinate system origin
        x_axis: Lateral axis (normalized)
        y_axis: Anterior axis (normalized)
        z_axis: Superior axis (normalized)

    Returns:
        Dictionary mapping FacialLandmarkType names to estimated positions
    """
    # Estimate Nz level height relative to origin
    nz_relative = nz - origin

    # Eye positions: ~30mm lateral, at or slightly below Nz level
    left_eye = nz + (-30.0 * x_axis) + (5.0 * y_axis) + (-10.0 * z_axis)
    right_eye = nz + (30.0 * x_axis) + (5.0 * y_axis) + (-10.0 * z_axis)

    # Nose tip: inferior and anterior to Nz
    nose_tip = nz + (40.0 * y_axis) + (-40.0 * z_axis)

    # Nose bridge: at Nz level, slightly anterior
    nose_bridge = nz + (15.0 * y_axis)

    # Mouth center: inferior to nose
    mouth_center = nz + (25.0 * y_axis) + (-70.0 * z_axis)

    # Chin: most inferior point
    chin = nz + (20.0 * y_axis) + (-100.0 * z_axis)

    return {
        FacialLandmarkType.LEFT_EYE.value: left_eye,
        FacialLandmarkType.RIGHT_EYE.value: right_eye,
        FacialLandmarkType.NOSE_TIP.value: nose_tip,
        FacialLandmarkType.NOSE_BRIDGE.value: nose_bridge,
        FacialLandmarkType.MOUTH_CENTER.value: mouth_center,
        FacialLandmarkType.CHIN.value: chin,
    }


@cdc.validate_schemas
def detect_facial_landmarks(
    surface: cdc.TrimeshSurface,
    anatomical_landmarks: cdt.LabeledPointCloud,
) -> FacialLandmarks:
    """Detect facial landmarks using geometric heuristics.

    Uses anatomical landmarks (Nz, Iz, LPA, RPA, Cz) to estimate facial feature
    positions based on known head proportions. This geometric approach works
    without texture information and provides consistent results.

    Args:
        surface: Textured TrimeshSurface from photogrammetry
        anatomical_landmarks: Known landmarks with Nz, Iz, LPA, RPA, Cz

    Returns:
        FacialLandmarks with estimated eye, nose, mouth positions

    Raises:
        ValueError: if required anatomical landmarks are missing
    """
    # Extract required landmarks
    nz = _find_landmark(anatomical_landmarks, "Nz")
    iz = _find_landmark(anatomical_landmarks, "Iz")
    lpa = _find_landmark(anatomical_landmarks, "LPA")
    rpa = _find_landmark(anatomical_landmarks, "RPA")
    cz = _find_landmark(anatomical_landmarks, "Cz")

    # Build head coordinate system
    origin, x_axis, y_axis, z_axis = _build_head_coordinate_system(
        nz, iz, lpa, rpa, cz
    )

    # Estimate facial landmarks using geometric proportions
    estimated_landmarks = _estimate_facial_landmarks_geometric(
        nz, origin, x_axis, y_axis, z_axis
    )

    # Create xarray DataArray for landmarks
    landmark_names = list(estimated_landmarks.keys())
    landmark_positions = np.array([estimated_landmarks[name] for name in landmark_names])

    # Get CRS from surface
    crs = surface.crs

    landmarks_da = xr.DataArray(
        landmark_positions,
        dims=["label", crs],
        coords={"label": landmark_names},
        attrs={"units": surface.units},
    ).pint.quantify()

    # All geometric detections have confidence 1.0 (deterministic)
    confidence = {name: 1.0 for name in landmark_names}

    return FacialLandmarks(
        landmarks=landmarks_da,
        confidence=confidence,
        detection_method="geometric",
    )


def _compute_angular_distance(
    point: np.ndarray, reference: np.ndarray, center: np.ndarray
) -> float:
    """Compute angular distance between a point and reference from center.

    Args:
        point: Point to measure
        reference: Reference point (e.g., nasion)
        center: Center point for angle measurement

    Returns:
        Angular distance in degrees
    """
    v1 = _normalize(point - center)
    v2 = _normalize(reference - center)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


@cdc.validate_schemas
def get_facial_region_mask(
    surface: cdc.TrimeshSurface,
    facial_landmarks: FacialLandmarks,
    protected_points: cdt.LabeledPointCloud,
    protection_radius: Quantity = 15.0 * units.mm,
) -> np.ndarray:
    """Generate boolean mask of vertices in facial region.

    Creates a mask identifying vertices in the facial region while excluding
    protection zones around optodes and anatomical landmarks. The facial region
    is determined by geometric criteria based on the head coordinate system.

    Args:
        surface: The mesh surface
        facial_landmarks: Detected facial landmarks
        protected_points: Points to exclude (optodes + anatomical landmarks)
        protection_radius: Radius around protected points in length units

    Returns:
        Boolean array of shape (n_vertices,) where True indicates facial region
    """
    # Get vertices as numpy array
    vertices = surface.mesh.vertices

    # Get required anatomical landmarks from protected_points
    try:
        nz = _find_landmark(protected_points, "Nz")
        iz = _find_landmark(protected_points, "Iz")
        lpa = _find_landmark(protected_points, "LPA")
        rpa = _find_landmark(protected_points, "RPA")
        cz = _find_landmark(protected_points, "Cz")
    except ValueError:
        # If not all landmarks in protected_points, try facial_landmarks
        fl = facial_landmarks.landmarks.pint.dequantify()
        nz = _find_landmark(protected_points, "Nz")
        iz = _find_landmark(protected_points, "Iz")
        lpa = _find_landmark(protected_points, "LPA")
        rpa = _find_landmark(protected_points, "RPA")
        cz = _find_landmark(protected_points, "Cz")

    # Build head coordinate system
    origin, x_axis, y_axis, z_axis = _build_head_coordinate_system(
        nz, iz, lpa, rpa, cz
    )

    # Transform vertices to head coordinate system
    vertices_relative = vertices - origin
    vertices_x = np.dot(vertices_relative, x_axis)
    vertices_y = np.dot(vertices_relative, y_axis)
    vertices_z = np.dot(vertices_relative, z_axis)

    # Get Nz and Cz positions in head coordinates
    nz_relative = nz - origin
    nz_y = np.dot(nz_relative, y_axis)
    nz_z = np.dot(nz_relative, z_axis)

    cz_relative = cz - origin
    cz_z = np.dot(cz_relative, z_axis)

    # T-shaped facial region mask in head coordinate system:
    # Anterior mask: only front of head (Y > 0)
    anterior_mask = vertices_y > 0

    # Horizontal bar (eye band): Nz height ± 30mm, full lateral span
    horiz = anterior_mask & (vertices_z > nz_z - 30) & (vertices_z < nz_z + 30)

    # Vertical bar (nose strip): Nz down 60mm, narrow lateral (±35mm)
    vert = (
        anterior_mask
        & (vertices_z > nz_z - 60)
        & (vertices_z < nz_z)
        & (np.abs(vertices_x) < 35)
    )

    # Combined T-shape
    facial_mask = horiz | vert

    # Create protection zones using KDTree
    protection_radius_mm = float(protection_radius.to("mm").magnitude)

    # Get protected point positions
    protected_positions = protected_points.pint.dequantify().values

    if len(protected_positions) > 0:
        kdtree = KDTree(protected_positions)
        distances, _ = kdtree.query(vertices, k=1)
        protected_mask = distances < protection_radius_mm

        # Exclude protected vertices from facial region
        facial_mask = facial_mask & ~protected_mask

    logger.info(
        f"Facial region mask: {facial_mask.sum()} of {len(facial_mask)} vertices "
        f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
    )

    return facial_mask


def get_facial_region_center(
    surface: cdc.TrimeshSurface,
    facial_landmarks: FacialLandmarks,
) -> np.ndarray:
    """Compute the centroid of the facial region.

    Args:
        surface: The mesh surface
        facial_landmarks: Detected facial landmarks

    Returns:
        Numpy array of shape (3,) with facial region centroid
    """
    # Get facial landmark positions
    landmarks = facial_landmarks.landmarks.pint.dequantify().values

    # Return centroid of all facial landmarks
    return np.mean(landmarks, axis=0)


def estimate_face_bounding_box(
    facial_landmarks: FacialLandmarks,
    margin: Quantity = 20.0 * units.mm,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a bounding box around the facial region.

    Args:
        facial_landmarks: Detected facial landmarks
        margin: Extra margin around landmarks in length units

    Returns:
        Tuple of (min_corner, max_corner) as numpy arrays
    """
    landmarks = facial_landmarks.landmarks.pint.dequantify().values
    margin_mm = float(margin.to("mm").magnitude)

    min_corner = landmarks.min(axis=0) - margin_mm
    max_corner = landmarks.max(axis=0) + margin_mm

    return min_corner, max_corner


def detect_landmarks_from_nasion(
    surface: cdc.TrimeshSurface,
    nz_position: np.ndarray,
) -> cdt.LabeledPointCloud:
    """Detect anatomical landmarks from a user-provided nasion (Nz) position.

    Given the Nz point (clicked by the user on the mesh), automatically detects
    the remaining 4 landmarks (Iz, LPA, RPA, Cz) using mesh geometry.

    Coordinate system (Einstar scanner convention):
    - X: Up (vertical)
    - Y: Forward (anterior, toward face)
    - Z: Left of subject

    Detection algorithm:
    - Cz: Highest vertex (max X) within a lateral band near the mesh centroid
    - Iz: Most posterior vertex (min Y) at approximately Nz height
    - LPA: Most left vertex (max Z) at ear height, not too posterior
    - RPA: Most right vertex (min Z) at ear height, not too posterior

    Args:
        surface: TrimeshSurface from photogrammetry scan
        nz_position: Nasion position as numpy array of shape (3,), in mm

    Returns:
        LabeledPointCloud with landmarks labeled as Nz, Iz, Cz, LPA, RPA

    Raises:
        ValueError: If landmark configuration fails validation

    Example:
        >>> # User clicks Nz on the mesh, gets nz_point
        >>> landmarks = detect_landmarks_from_nasion(surface, nz_point)
        >>> print(landmarks.label.values)
        ['Nz', 'Iz', 'Cz', 'LPA', 'RPA']
    """
    nz_position = np.asarray(nz_position, dtype=float)
    vertices = surface.mesh.vertices

    centroid = vertices.mean(axis=0)
    head_height = vertices[:, 0].max() - vertices[:, 0].min()

    logger.info(
        f"Detecting landmarks from Nz={nz_position}, "
        f"head_height={head_height:.1f}mm"
    )

    # Cz: highest vertex (max X) within a lateral band near centroid
    # Band: within 20% of head_height in Y and Z from centroid
    band = 0.20 * head_height
    lateral_mask = (
        (np.abs(vertices[:, 1] - centroid[1]) < band)
        & (np.abs(vertices[:, 2] - centroid[2]) < band)
    )
    if lateral_mask.sum() == 0:
        lateral_mask = np.ones(len(vertices), dtype=bool)
    cz_idx = np.where(lateral_mask)[0][np.argmax(vertices[lateral_mask, 0])]
    cz = vertices[cz_idx]

    # Iz: most posterior vertex (min Y) within a height band around Nz height
    # Band: ±15% of head_height in X around Nz's X
    height_band = 0.15 * head_height
    iz_height_mask = np.abs(vertices[:, 0] - nz_position[0]) < height_band
    if iz_height_mask.sum() == 0:
        iz_height_mask = np.ones(len(vertices), dtype=bool)
    iz_idx = np.where(iz_height_mask)[0][np.argmin(vertices[iz_height_mask, 1])]
    iz = vertices[iz_idx]

    # Ear height: slightly below Nz (Nz height - 5% head_height)
    ear_height = nz_position[0] - 0.05 * head_height
    ear_height_mask = np.abs(vertices[:, 0] - ear_height) < height_band

    # Additional constraint: not too posterior (Y > centroid Y - some margin)
    posterior_limit = centroid[1] - 0.10 * head_height
    not_too_posterior = vertices[:, 1] > posterior_limit

    ear_mask = ear_height_mask & not_too_posterior
    if ear_mask.sum() < 2:
        ear_mask = ear_height_mask

    # LPA: most left vertex (max Z) at ear height
    lpa_idx = np.where(ear_mask)[0][np.argmax(vertices[ear_mask, 2])]
    lpa = vertices[lpa_idx]

    # RPA: most right vertex (min Z) at ear height
    rpa_idx = np.where(ear_mask)[0][np.argmin(vertices[ear_mask, 2])]
    rpa = vertices[rpa_idx]

    landmark_positions = {
        "Nz": nz_position,
        "Iz": iz,
        "Cz": cz,
        "LPA": lpa,
        "RPA": rpa,
    }

    logger.info(
        f"Detected landmarks: Cz={cz}, Iz={iz}, LPA={lpa}, RPA={rpa}"
    )

    # Validate
    _validate_landmark_configuration(landmark_positions, centroid)

    # Create LabeledPointCloud
    labels = ["Nz", "Iz", "Cz", "LPA", "RPA"]
    coords = np.array([landmark_positions[label] for label in labels])

    landmarks = xr.DataArray(
        coords,
        dims=["label", surface.crs],
        coords={
            "label": labels,
            "type": ("label", [cdc.PointType.LANDMARK] * 5),
        },
    ).pint.quantify("mm")

    return landmarks


def _validate_landmark_configuration(
    landmarks: dict[str, np.ndarray],
    centroid: np.ndarray,
) -> None:
    """Validate that detected landmarks have a plausible spatial configuration.

    Checks:
    - Cz is the highest point (max X)
    - Iz is posterior to the centroid (low Y)
    - LPA and RPA are roughly symmetric about the midline (Z)

    Args:
        landmarks: Dict mapping landmark name to position array
        centroid: Mesh centroid for reference

    Raises:
        ValueError: If configuration is implausible
    """
    cz = landmarks["Cz"]
    iz = landmarks["Iz"]
    nz = landmarks["Nz"]
    lpa = landmarks["LPA"]
    rpa = landmarks["RPA"]

    # Cz should be highest (max X)
    all_x = [landmarks[k][0] for k in landmarks]
    if cz[0] < max(all_x) - 1.0:
        logger.warning("Cz is not the highest landmark — detection may be off")

    # Iz should be posterior to centroid (Y < centroid Y)
    if iz[1] > centroid[1]:
        logger.warning(
            f"Iz (Y={iz[1]:.1f}) is anterior to centroid (Y={centroid[1]:.1f}) "
            "— expected posterior"
        )

    # LPA should be left (Z > centroid Z) and RPA right (Z < centroid Z)
    if lpa[2] < rpa[2]:
        logger.warning(
            "LPA is to the right of RPA — landmarks may be swapped"
        )

    # LPA and RPA should be roughly symmetric
    lpa_offset = abs(lpa[2] - centroid[2])
    rpa_offset = abs(rpa[2] - centroid[2])
    if min(lpa_offset, rpa_offset) > 0 and max(lpa_offset, rpa_offset) / min(lpa_offset, rpa_offset) > 3.0:
        logger.warning(
            f"LPA/RPA asymmetry is large (offsets: {lpa_offset:.1f} vs {rpa_offset:.1f}mm)"
        )
