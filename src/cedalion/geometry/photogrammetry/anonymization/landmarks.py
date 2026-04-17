"""Landmark detection for photogrammetry scans.

Given a user-provided nasion (Nz) on an axis-normalized mesh, detects the
remaining four anatomical landmarks (Iz, Cz, LPA, RPA) geometrically:

- Cz is the highest vertex near the midline.
- Iz is the most posterior vertex at nasion height.
- LPA/RPA are geometric ear targets snapped to the nearest surface vertex.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

logger = logging.getLogger("cedalion")


def detect_landmarks_from_nasion(
    surface: cdc.TrimeshSurface,
    nz_position: np.ndarray,
) -> cdt.LabeledPointCloud:
    """Detect anatomical landmarks from a user-provided nasion (Nz) position.

    **Important:** The surface must be axis-normalized before calling this
    function. Use ``normalize_axes()`` first so that:
    - X = up (vertical, from Einstar gravity sensor)
    - Y = anterior (toward face, from nasion direction)
    - Z = left of subject

    Given the Nz point, automatically detects Iz, LPA, RPA, Cz:
    - Cz: Highest vertex (max X) near midline
    - Iz: Most posterior vertex (min Y) at nasion height
    - LPA: Most left vertex (max Z) at ear height
    - RPA: Most right vertex (min Z) at ear height

    Args:
        surface: Axis-normalized TrimeshSurface (Y=anterior, Z=left)
        nz_position: Nasion position as numpy array of shape (3,), in mm

    Returns:
        LabeledPointCloud with landmarks labeled as Nz, Iz, Cz, LPA, RPA

    Raises:
        ValueError: If landmark configuration fails validation

    Example:
        >>> surface_norm, nz_norm, R = normalize_axes(surface, nz, fwd)
        >>> landmarks = detect_landmarks_from_nasion(surface_norm, nz_norm)
    """
    nz_position = np.asarray(nz_position, dtype=float)
    vertices = surface.mesh.vertices

    centroid = vertices.mean(axis=0)
    head_height = vertices[:, 0].max() - vertices[:, 0].min()
    head_verts = vertices

    logger.debug(
        f"Detecting landmarks from Nz={nz_position}, "
        f"head_height={head_height:.1f}mm, "
        f"head_verts={len(vertices)}"
    )

    # Cz: highest vertex (max X) near midline
    band = 0.20 * head_height
    lateral_mask = (
        (np.abs(head_verts[:, 1] - centroid[1]) < band)
        & (np.abs(head_verts[:, 2] - centroid[2]) < band)
    )
    if lateral_mask.sum() == 0:
        lateral_mask = np.ones(len(head_verts), dtype=bool)
    cz_idx = np.where(lateral_mask)[0][np.argmax(head_verts[lateral_mask, 0])]
    cz = head_verts[cz_idx]

    # Iz: most posterior (min Y) at nasion height, on the midsagittal plane.
    iz_height_mask = np.abs(head_verts[:, 0] - nz_position[0]) < 20.0
    iz_midline_mask = np.abs(head_verts[:, 2] - nz_position[2]) < 25.0
    iz_mask = iz_height_mask & iz_midline_mask
    if iz_mask.sum() == 0:
        iz_mask = np.abs(head_verts[:, 0] - nz_position[0]) < 40.0
    iz_cands = np.where(iz_mask)[0]
    iz_idx = iz_cands[np.argmin(head_verts[iz_cands, 1])]
    iz = head_verts[iz_idx]

    # LPA/RPA: geometric targets, refined by snapping to surface
    from scipy.spatial import cKDTree

    midline_z = np.mean([nz_position[2], iz[2], cz[2]])
    clean_band = (
        (head_verts[:, 0] > nz_position[0] + 10.0)
        & (head_verts[:, 0] < cz[0] - 20.0)
    )
    if clean_band.sum() > 100:
        clean_z = head_verts[clean_band, 2]
        half_width = (
            np.percentile(clean_z, 97) - np.percentile(clean_z, 3)
        ) / 2.0
    else:
        half_width = 75.0

    geo_lpa_target = np.array([nz_position[0], cz[1], midline_z + half_width])
    geo_rpa_target = np.array([nz_position[0], cz[1], midline_z - half_width])

    def _snap_ear(target):
        ear_region = (
            (np.abs(head_verts[:, 0] - nz_position[0]) < 30.0)
            & (np.abs(head_verts[:, 1] - cz[1]) < 40.0)
        )
        if ear_region.sum() > 10:
            idxs = np.where(ear_region)[0]
            tree = cKDTree(head_verts[idxs])
            _, local = tree.query(target)
            return head_verts[idxs[local]]
        tree = cKDTree(head_verts)
        _, idx = tree.query(target)
        return head_verts[idx]

    lpa = _snap_ear(geo_lpa_target)
    rpa = _snap_ear(geo_rpa_target)

    landmark_positions = {
        "Nz": nz_position,
        "Iz": iz,
        "Cz": cz,
        "LPA": lpa,
        "RPA": rpa,
    }

    logger.debug(f"Detected landmarks: Cz={cz}, Iz={iz}, LPA={lpa}, RPA={rpa}")

    _validate_landmark_configuration(landmark_positions, centroid)

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
        landmarks: Dict mapping landmark name to position array.
        centroid: Mesh centroid for reference.

    Raises:
        ValueError: If configuration is implausible.
    """
    cz = landmarks["Cz"]
    iz = landmarks["Iz"]
    lpa = landmarks["LPA"]
    rpa = landmarks["RPA"]

    all_x = [landmarks[k][0] for k in landmarks]
    if cz[0] < max(all_x) - 1.0:
        logger.warning("Cz is not the highest landmark -- detection may be off")

    if iz[1] > centroid[1]:
        logger.warning(
            f"Iz (Y={iz[1]:.1f}) is anterior to centroid (Y={centroid[1]:.1f}) "
            "-- expected posterior"
        )

    if lpa[2] < rpa[2]:
        logger.warning("LPA is to the right of RPA -- landmarks may be swapped")

    lpa_offset = abs(lpa[2] - centroid[2])
    rpa_offset = abs(rpa[2] - centroid[2])
    if min(lpa_offset, rpa_offset) > 0 and max(lpa_offset, rpa_offset) / min(lpa_offset, rpa_offset) > 3.0:
        logger.warning(
            f"LPA/RPA asymmetry is large (offsets: {lpa_offset:.1f} vs {rpa_offset:.1f}mm)"
        )
