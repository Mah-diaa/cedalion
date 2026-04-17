"""Facial-region mask construction and application.

Given the 5 aligned landmarks, builds a boolean deletion mask and applies
it to the mesh:

- ``detect_cap_boundary`` finds the X height where the EEG cap front edge
  sits so the mask can be clamped below it.
- ``face_mask_from_landmarks`` unions a forward face region with two ear
  spheres, clamped below the cap.
- ``delete_masked_vertices`` drops triangles touching any masked vertex.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np

import cedalion.dataclasses as cdc

logger = logging.getLogger("cedalion")


def detect_cap_boundary(
    verts: np.ndarray,
    Nz: np.ndarray,
    Cz: np.ndarray,
    ear_mid: np.ndarray,
    mid_z: float,
    band_width: float = 15.0,
    bin_size: float = 2.0,
    foot_grad_threshold: float = 0.2,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Find the X height where the EEG cap front edge sits.

    Scans upward from Nz along the midline and records max-Y per X-bin. The cap
    protrudes in +Y, so Y-max along the midline lives on the cap. The cap edge
    is the foot of the steep rise leading to the peak: walk back from the peak
    until the smoothed gradient drops below ``foot_grad_threshold``.

    Expects the aligned frame (X=up, Y=anterior, Z=left).

    Args:
        verts: Mesh vertices, shape (N, 3).
        Nz: Nasion position.
        Cz: Cz position.
        ear_mid: Midpoint of Lpa/Rpa.
        mid_z: Midline Z value (e.g. ``0.5 * (Lpa[2] + Rpa[2])``).
        band_width: Z-band half-width for the midline Y-profile (mm).
        bin_size: X-bin size for the Y-profile (mm).
        foot_grad_threshold: dY/dX below this value marks the foot of the rise.

    Returns:
        Tuple of (cap_x, profile_x, profile_y_raw, profile_y_smooth).
    """
    from scipy.signal import savgol_filter

    in_band = np.abs(verts[:, 2] - mid_z) < band_width
    above_nz = verts[:, 0] > Nz[0]
    anterior = verts[:, 1] > ear_mid[1]
    sel_verts = verts[in_band & above_nz & anterior]

    bins = np.arange(Nz[0], Cz[0], bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    max_y = np.full(len(bin_centers), np.nan)

    for i in range(len(bin_centers)):
        in_bin = (sel_verts[:, 0] >= bins[i]) & (sel_verts[:, 0] < bins[i + 1])
        if in_bin.any():
            max_y[i] = sel_verts[in_bin, 1].max()

    valid = ~np.isnan(max_y)
    if valid.sum() < 7:
        fallback = 0.5 * (Nz[0] + Cz[0])
        return fallback, bin_centers[valid], max_y[valid], max_y[valid]

    xv = bin_centers[valid]
    yv = max_y[valid]

    win = min(11, len(yv) if len(yv) % 2 == 1 else len(yv) - 1)
    win = max(win, 5)
    yv_s = savgol_filter(yv, window_length=win, polyorder=2)

    peak_idx = int(np.argmax(yv_s))
    grad = np.gradient(yv_s, xv)
    cap_x = xv[0]
    for i in range(peak_idx - 1, 0, -1):
        if grad[i] < foot_grad_threshold:
            cap_x = xv[i]
            break

    return cap_x, xv, yv, yv_s


def face_mask_from_landmarks(
    verts: np.ndarray,
    Nz: np.ndarray,
    Iz: np.ndarray,
    Cz: np.ndarray,
    Lpa: np.ndarray,
    Rpa: np.ndarray,
    cap_x: float | None = None,
    ear_delete_radius: float = 40.0,
) -> tuple[np.ndarray, dict]:
    """Build face + ear deletion mask from the 5 landmarks (aligned frame).

    Mask is the union of two regions, both clamped below the cap boundary:

    1. Face region: anterior to the ear coronal plane (Y > ear_mid_Y).
    2. Ear spheres: ``ear_delete_radius`` mm around Lpa and Rpa.

    Expects the aligned frame (X=up, Y=anterior, Z=left).

    Args:
        verts: Mesh vertices, shape (N, 3).
        Nz, Iz, Cz, Lpa, Rpa: 5 landmark positions in the aligned frame.
        cap_x: Upper bound X value (typically from ``detect_cap_boundary``).
            Defaults to Nz[0] if not provided.
        ear_delete_radius: Sphere radius around Lpa/Rpa in mm.

    Returns:
        Tuple of (mask, info). ``mask`` is a boolean array of shape (N,).
        ``info`` has keys ``upper_bound``, ``ear_mid``, and ``counts``
        (per-region vertex counts).
    """
    ear_mid = 0.5 * (Lpa + Rpa)
    upper_bound = cap_x if cap_x is not None else Nz[0]

    below_cap = verts[:, 0] < upper_bound
    anterior = verts[:, 1] > ear_mid[1]
    face_region = below_cap & anterior

    d_lpa = np.linalg.norm(verts - Lpa, axis=1)
    d_rpa = np.linalg.norm(verts - Rpa, axis=1)
    ear_region = ((d_lpa < ear_delete_radius) | (d_rpa < ear_delete_radius)) & below_cap

    mask = face_region | ear_region

    info = {
        "upper_bound": upper_bound,
        "ear_mid": ear_mid,
        "counts": {
            "below_cap": int(below_cap.sum()),
            "face_region": int(face_region.sum()),
            "ear_region": int(ear_region.sum()),
            "all": int(mask.sum()),
        },
    }
    return mask, info


def delete_masked_vertices(
    surface: cdc.TrimeshSurface,
    mask: np.ndarray,
) -> cdc.TrimeshSurface:
    """Drop triangles touching any masked vertex and strip unreferenced vertices.

    Args:
        surface: Input TrimeshSurface.
        mask: Boolean array of shape (n_vertices,). True = vertex to remove.

    Returns:
        New TrimeshSurface with masked vertices (and the faces touching them)
        removed. CRS and units are preserved.
    """
    mesh_copy = surface.mesh.copy()
    faces_to_remove = mask[mesh_copy.faces].any(axis=1)
    mesh_copy.update_faces(~faces_to_remove)
    mesh_copy.remove_unreferenced_vertices()
    return cdc.TrimeshSurface(
        mesh=mesh_copy, crs=surface.crs, units=surface.units,
    )
