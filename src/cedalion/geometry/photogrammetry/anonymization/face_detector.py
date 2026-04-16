"""Facial region detection for photogrammetry scans.

Provides axis normalization, head isolation, landmark detection from a
user-provided nasion (Nz), and facial region mask generation. Supports both
a MediaPipe contour path (via ``get_facial_region_mask_from_nasion``) and a
pure-geometric landmark-only path (``align_axes_from_landmarks`` +
``detect_cap_boundary`` + ``face_mask_from_landmarks``).

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
from scipy.spatial import KDTree
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units

logger = logging.getLogger("cedalion")


def normalize_axes(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    forward_direction: np.ndarray = None,
) -> tuple[cdc.TrimeshSurface, np.ndarray, np.ndarray]:
    """Rotate mesh around X-axis so Y=anterior (toward face), Z=left.

    The Einstar scanner has X=up (gravity-based) but Y/Z are arbitrary per
    scan session. This function uses the nasion position to determine the
    forward direction and rotates the mesh so that Y consistently points
    toward the face (anterior) and Z points left.

    Args:
        surface: TrimeshSurface in raw Einstar coordinates.
        nasion: Nasion position as numpy array of shape (3,).
        forward_direction: Optional pre-computed forward unit vector in YZ
            plane (X=0). If None, computed from nasion vs upper-head centroid.

    Returns:
        Tuple of (rotated_surface, rotated_nasion, rotation_matrix).
        rotation_matrix is 3x3 and can be applied to other points.
    """
    vertices = surface.mesh.vertices

    if forward_direction is None:
        # Compute forward direction: nasion minus upper-head centroid in YZ
        x_max = vertices[:, 0].max()
        x_min = vertices[:, 0].min()
        upper = vertices[vertices[:, 0] > x_min + 0.6 * (x_max - x_min)]
        centroid_yz = np.array([0.0, upper[:, 1].mean(), upper[:, 2].mean()])
        nasion_yz = np.array([0.0, nasion[1], nasion[2]])
        forward_direction = nasion_yz - centroid_yz
        fwd_norm = np.linalg.norm(forward_direction)
        if fwd_norm < 1e-6:
            logger.warning("Cannot compute forward direction — returning unchanged")
            return surface, nasion.copy(), np.eye(3)
        forward_direction = forward_direction / fwd_norm

    # Angle of forward_direction relative to +Y in the YZ plane
    angle = np.arctan2(forward_direction[2], forward_direction[1])

    # Rotation around X-axis by -angle to align forward with +Y
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])

    # Rotate vertices
    rotated_verts = vertices @ R.T

    # Build new mesh with rotated vertices, same faces and texture
    import trimesh
    new_mesh = trimesh.Trimesh(
        vertices=rotated_verts,
        faces=surface.mesh.faces,
        visual=surface.mesh.visual,
        process=False,
    )
    rotated_surface = cdc.TrimeshSurface(new_mesh, crs=surface.crs, units=surface.units)

    # Rotate nasion
    rotated_nasion = R @ nasion

    logger.debug(
        f"Axis normalization: rotated {np.degrees(angle):.1f}deg around X. "
        f"Y now points anterior."
    )

    return rotated_surface, rotated_nasion, R


def _largest_component_mask(mesh) -> np.ndarray:
    """Boolean vertex mask selecting the largest connected component.

    Einstar scans are non-watertight and frequently contain tiny floating
    mesh fragments (loose triangles, cable shreds, background patches)
    that sit far outside the head. Any step that uses vertex extrema
    (e.g. head sphere centroid, lateral-widest heuristics) will be
    dragged off into empty space by those fragments. Stripping them
    up-front prevents that.

    Uses ``trimesh.graph.connected_component_labels`` on face adjacency
    rather than ``mesh.split(only_watertight=False)`` — split allocates
    one full ``Trimesh`` per component and can OOM when a scan has
    thousands of tiny fragments. Labels give one int per face with
    zero per-component allocation.

    Args:
        mesh: A ``trimesh.Trimesh`` instance.

    Returns:
        Boolean array of shape ``(n_vertices,)``. All-True if the mesh
        is empty or already a single connected component.
    """
    import trimesh
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.ones(n_verts, dtype=bool)

    # Einstar OBJs duplicate vertices along UV seams for texturing. The
    # default trimesh.Trimesh.merge_vertices() does NOT merge across UV
    # seams (it preserves texture), so face adjacency on the raw mesh
    # (or on a merge_vertices'd copy) over-fragments the head into
    # thousands of islands. We need POSITION-only merging for
    # connectivity analysis: unique_rows on vertex coordinates gives a
    # canonical-id mapping that heals the seams.
    unique_idx, inverse = trimesh.grouping.unique_rows(
        np.asarray(mesh.vertices)
    )
    canonical_faces = inverse[mesh.faces]
    adjacency = trimesh.graph.face_adjacency(faces=canonical_faces)
    face_labels = trimesh.graph.connected_component_labels(
        adjacency, node_count=n_faces
    )
    counts = np.bincount(face_labels)
    if len(counts) <= 1:
        return np.ones(n_verts, dtype=bool)
    biggest_label = int(np.argmax(counts))
    face_mask = face_labels == biggest_label

    # Apply face mask to the ORIGINAL faces -> original vertex indices
    # (so colors/UVs/normals attached to the caller's mesh stay valid).
    kept_vidx = np.unique(mesh.faces[face_mask])
    mask = np.zeros(n_verts, dtype=bool)
    mask[kept_vidx] = True
    return mask


def isolate_head(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    radius: float = 220.0,
) -> tuple[cdc.TrimeshSurface, np.ndarray]:
    """Remove shoulders, body, and chair — keep only the head.

    Uses a sphere centered on the upper-head centroid. The radius is
    chosen to capture the full head (~180mm wide x 230mm tall) while
    excluding shoulders and body. Scans that are already head-only are
    returned unchanged (the sphere just contains everything).

    The surface must be axis-normalized first (X=up, Y=anterior, Z=left).

    Args:
        surface: Axis-normalized TrimeshSurface.
        nasion: Nasion position as numpy array of shape (3,).
        radius: Sphere radius in mm (default 220). A human head has
            ~90mm radius; 220mm adds margin for ears and jaw.

    Returns:
        Tuple of (head_surface, head_mask). head_mask is a boolean
        array of shape (n_vertices,) indicating which original
        vertices were kept.
    """
    import trimesh

    vertices = surface.mesh.vertices
    faces = surface.mesh.faces

    # Sphere center: Y/Z from upper-head centroid (good lateral centering).
    # X: use the lower of (upper centroid, midpoint of top+nasion).
    x_max = vertices[:, 0].max()
    x_min = vertices[:, 0].min()
    upper = vertices[:, 0] > x_min + 0.6 * (x_max - x_min)
    center = vertices[upper].mean(axis=0)
    midpoint_x = (x_max + nasion[0]) / 2.0
    center[0] = min(center[0], midpoint_x)

    # Sphere mask
    dist = np.linalg.norm(vertices - center, axis=1)
    head_mask = dist < radius

    # Always strip disconnected fragments (floating triangles, cables,
    # background patches). No-op on clean single-component scans.
    head_mask = head_mask & _largest_component_mask(surface.mesh)

    # If the sphere captures almost everything, skip trimming
    if head_mask.sum() < 100 or head_mask.mean() > 0.95:
        logger.debug(
            f"Head isolation: sphere contains {head_mask.mean()*100:.0f}% "
            f"of vertices — scan is already head-only"
        )
        return surface, head_mask

    # Build new mesh with only head faces
    # A face is kept if ALL its vertices are in the head
    face_mask = head_mask[faces].all(axis=1)
    head_faces = faces[face_mask]

    # Reindex vertices: only keep referenced vertices
    kept_verts = np.unique(head_faces)
    reindex = np.full(len(vertices), -1, dtype=int)
    reindex[kept_verts] = np.arange(len(kept_verts))
    new_faces = reindex[head_faces]
    new_verts = vertices[kept_verts]

    # Transfer visual — subset vertex colors to match kept vertices
    new_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )
    try:
        old_visual = surface.mesh.visual
        if hasattr(old_visual, 'vertex_colors') and len(old_visual.vertex_colors) == len(vertices):
            new_mesh.visual.vertex_colors = old_visual.vertex_colors[kept_verts]
    except Exception:
        pass

    head_surface = cdc.TrimeshSurface(
        new_mesh, crs=surface.crs, units=surface.units
    )

    logger.debug(
        f"Head isolation: {len(vertices):,} -> {len(new_verts):,} vertices "
        f"({len(vertices) - len(new_verts):,} removed), "
        f"center=[{center[0]:.0f},{center[1]:.0f},{center[2]:.0f}], "
        f"radius={radius:.0f}mm"
    )

    return head_surface, head_mask


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
        """Snap target to nearest surface vertex in ear region."""
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

    logger.debug(
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


def get_facial_region_mask_from_nasion(
    surface: cdc.TrimeshSurface,
    nz: np.ndarray,
    forward_direction: np.ndarray,
    face_contour_3d: np.ndarray | None = None,
    protected_points: cdt.LabeledPointCloud = None,
    protection_radius: Quantity = 15.0 * units.mm,
    lateral_extension: float = 70.0,
    lower_width_scale: float = 2.5,
) -> np.ndarray:
    """Generate facial region mask using nasion and MediaPipe face contour.

    This function is axis-independent -- it uses the forward direction and
    face contour from MediaPipe rather than relying on axis alignment.
    It does NOT depend on LPA/RPA accuracy.

    The mask uses a two-part strategy derived entirely from the contour:
    - **Oval**: The MediaPipe face contour polygon defines precise
      forehead and jawline boundaries (with a depth filter to prevent
      top-of-head bleed).
    - **Side band**: 3D proximity to contour points -- any vertex within
      ``lateral_extension`` mm of a contour point is included, vertically
      constrained to chin-to-temple height to avoid top-of-head bleed.
      This naturally follows the head curvature and covers the ears.

    When no face contour is available (manual nasion mode), falls back to
    a forward-facing hemisphere around the nasion.

    Args:
        surface: The mesh surface (axis-normalized or not)
        nz: Nasion position as numpy array of shape (3,)
        forward_direction: Unit vector pointing toward the face
        face_contour_3d: 3D face oval points from MediaPipe back-projection,
            shape (N, 3) with N >= 20, or None for hemisphere fallback
        protected_points: Points to exclude (optodes + anatomical landmarks)
        protection_radius: Radius around protected points
        lateral_extension: 3D proximity radius in mm from contour points
            for ear/temple coverage (default 70.0)
        lower_width_scale: Factor to widen the lower portion of the face
            contour (below nasion). 1.0 = no change, 2.5 = 150% wider at
            chin level. Smoothly blends from 1.0 at nasion to this value
            at chin. Keeps forehead border unchanged. (default 2.5)

    Returns:
        Boolean array of shape (n_vertices,) where True = facial region
    """
    from matplotlib.path import Path

    vertices = surface.mesh.vertices
    nz = np.asarray(nz, dtype=float)
    fwd = np.asarray(forward_direction, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)

    if face_contour_3d is not None and len(face_contour_3d) >= 20:
        # --- MediaPipe face oval approach (spherical projection) ---
        up_hint = np.array([1.0, 0.0, 0.0])  # X = up in Einstar coords
        if abs(np.dot(fwd, up_hint)) > 0.9:
            up_hint = np.array([0.0, 0.0, 1.0])
        u_axis = np.cross(fwd, up_hint)
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(fwd, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)

        # Head center: ~80mm behind nasion (roughly center of skull)
        head_center = nz - fwd * 80.0

        # Project contour onto unit sphere -> 2D
        c_dirs = face_contour_3d - head_center
        c_dirs = c_dirs / np.linalg.norm(c_dirs, axis=1, keepdims=True)
        contour_2d = np.column_stack([c_dirs @ u_axis, c_dirs @ v_axis])

        # Project vertices onto unit sphere -> 2D
        v_dirs = vertices - head_center
        v_fwd = v_dirs @ fwd  # positive = face side of head
        v_dirs = v_dirs / np.linalg.norm(v_dirs, axis=1, keepdims=True)
        verts_2d = np.column_stack([v_dirs @ u_axis, v_dirs @ v_axis])

        # Face mask = forward-facing vertices inside the contour polygon
        polygon = Path(contour_2d)
        facial_mask = polygon.contains_points(verts_2d) & (v_fwd > 0)

        logger.debug(
            f"Contour mask: {facial_mask.sum()} vertices "
            f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
        )
    else:
        # --- Fallback: forward-facing hemisphere ---
        verts_rel = vertices - nz
        fwd_dist = verts_rel @ fwd
        dist_to_nz = np.linalg.norm(verts_rel, axis=1)

        facial_mask = (fwd_dist > -20.0) & (dist_to_nz < 120.0)

        logger.debug(
            f"Hemisphere fallback mask: {facial_mask.sum()} vertices "
            f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
        )

    # Exclude protection zones
    protection_radius_mm = float(protection_radius.to("mm").magnitude)
    if protected_points is not None:
        protected_positions = protected_points.pint.dequantify().values
        if len(protected_positions) > 0:
            kdtree = KDTree(protected_positions)
            distances, _ = kdtree.query(vertices, k=1)
            protected_mask = distances < protection_radius_mm
            facial_mask = facial_mask & ~protected_mask

    logger.debug(
        f"Final facial mask: {facial_mask.sum()} of {len(facial_mask)} vertices "
        f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
    )

    return facial_mask


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


def align_axes_from_landmarks(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPointCloud,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPointCloud, np.ndarray]:
    """Derive full rotation from 5 landmarks and apply to mesh and landmarks.

    ``normalize_axes()`` only rotates around X. This function uses the full 5
    landmark set to align the head to the canonical frame:

        Z = lateral (Lpa - Rpa, pointing left)
        Y = anterior (Nz - ear_mid, orthogonal to Z)
        X = up (cross(Y, Z))

    Sign checks ensure Cz points +X and Nz points +Y.

    Args:
        surface: Axis-normalized TrimeshSurface (post ``normalize_axes`` and
            ``isolate_head``).
        landmarks: LabeledPointCloud with labels Nz, Iz, Cz, Lpa, Rpa
            (matching the surface frame).

    Returns:
        Tuple of (aligned_surface, aligned_landmarks, rotation_matrix).
        ``rotation_matrix`` is 3x3 and maps input-frame vectors to the aligned
        frame (apply as ``v @ R.T``).
    """
    import trimesh

    lm = landmarks.pint.dequantify().values
    labels = list(landmarks["label"].values)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    required = {"Nz", "Iz", "Cz", "Lpa", "Rpa"}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"Missing landmarks for alignment: {missing}")

    Nz = lm[idx["Nz"]]
    Cz = lm[idx["Cz"]]
    Lpa = lm[idx["Lpa"]]
    Rpa = lm[idx["Rpa"]]
    ear_mid = 0.5 * (Lpa + Rpa)

    z_ax = Lpa - Rpa
    z_ax = z_ax / np.linalg.norm(z_ax)

    nz_dir = Nz - ear_mid
    nz_dir = nz_dir - np.dot(nz_dir, z_ax) * z_ax
    y_ax = nz_dir / np.linalg.norm(nz_dir)

    x_ax = np.cross(y_ax, z_ax)

    if np.dot(Cz - ear_mid, x_ax) < 0:
        x_ax = -x_ax
    if np.dot(Nz - ear_mid, y_ax) < 0:
        y_ax = -y_ax
    z_ax = np.cross(x_ax, y_ax)

    R = np.vstack([x_ax, y_ax, z_ax])

    aligned_verts = np.asarray(surface.mesh.vertices) @ R.T
    new_mesh = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=surface.mesh.faces,
        visual=surface.mesh.visual,
        process=False,
    )
    aligned_surface = cdc.TrimeshSurface(
        new_mesh, crs=surface.crs, units=surface.units,
    )

    aligned_lm = lm @ R.T
    aligned_landmarks = landmarks.pint.dequantify().copy(data=aligned_lm).pint.quantify()

    return aligned_surface, aligned_landmarks, R


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


def eye_plane_rotation_matrix(
    r_eye_3d: np.ndarray,
    l_eye_3d: np.ndarray,
    forward_dir: np.ndarray,
) -> np.ndarray:
    """Build a rotation that sends the eye line to +Z and the forward ref to +Y.

    Rows of the returned matrix are the new basis vectors expressed in the old
    frame: ``p_new = R @ (p_old - center) + center``. The third row (Z) is the
    normalized eye line (L - R); the second row (Y) is the component of
    ``forward_dir`` orthogonal to the eye line; the first row (X) is ``Y x Z``
    so the result is head-up and right-handed.

    Typically fed MediaPipe eye landmarks plus the MediaPipe-derived forward
    direction, but it is pure geometry and works with any eye-line + forward
    reference.

    Args:
        r_eye_3d: Right eye position, shape (3,).
        l_eye_3d: Left eye position, shape (3,).
        forward_dir: Approximate anterior direction, shape (3,).

    Returns:
        3x3 rotation matrix.
    """
    eye_dir = np.asarray(l_eye_3d - r_eye_3d, dtype=float)
    z_new = eye_dir / np.linalg.norm(eye_dir)

    fref = np.asarray(forward_dir, dtype=float)
    fref = fref / np.linalg.norm(fref)
    y_new = fref - np.dot(fref, z_new) * z_new
    y_new = y_new / np.linalg.norm(y_new)

    x_new = np.cross(y_new, z_new)
    x_new = x_new / np.linalg.norm(x_new)
    return np.vstack([x_new, y_new, z_new])


def heuristic_lpa_rpa(
    head_verts: np.ndarray,
    eye_x: float,
    centroid_y: float,
    x_tol: float = 20.0,
    x_offset: float = -15.0,
    y_tol: float = 50.0,
    y_forward: float = 35.0,
    radius: float = 15.0,
) -> tuple[dict | None, dict | None]:
    """Approximate LPA/RPA as uncertainty spheres (not single points).

    Slices a horizontal band at roughly ear height (``eye_x + x_offset``), then
    picks the max-Z vertex as LPA center and min-Z as RPA center. Centers are
    nudged forward by ``y_forward`` to account for the heuristic's tendency to
    land behind the tragus.

    Expects ``head_verts`` in the eye-plane canonical frame (X=up, Y=anterior,
    Z=left). Returns ``(None, None)`` if fewer than 10 candidates are in the
    band.

    Args:
        head_verts: Mesh vertices in eye-plane frame, shape (N, 3).
        eye_x: X value of the eye line (average of both eyes).
        centroid_y: Head centroid Y, for Y-band selection.
        x_tol: X half-width of the ear-height band (mm).
        x_offset: Offset from ``eye_x`` to ear line (mm, negative = below eyes).
        y_tol: Y half-width of the ear-region band (mm).
        y_forward: Forward correction applied to both centers (mm).
        radius: Uncertainty-sphere radius (mm).

    Returns:
        Tuple of ``(lpa_dict, rpa_dict)``, each with keys ``center`` (np.ndarray,
        shape (3,)) and ``radius`` (float). ``(None, None)`` if the band has
        too few candidates.
    """
    x_band = np.abs(head_verts[:, 0] - (eye_x + x_offset)) < x_tol
    y_band = np.abs(head_verts[:, 1] - centroid_y) < y_tol
    mask = x_band & y_band
    candidates = head_verts[mask]
    if len(candidates) < 10:
        return None, None

    lpa_center = candidates[int(np.argmax(candidates[:, 2]))].astype(float).copy()
    rpa_center = candidates[int(np.argmin(candidates[:, 2]))].astype(float).copy()
    lpa_center[1] += y_forward
    rpa_center[1] += y_forward

    return (
        {"center": lpa_center, "radius": radius},
        {"center": rpa_center, "radius": radius},
    )


def mediapipe_face_mask_from_contour(
    verts: np.ndarray,
    nasion: np.ndarray,
    lpa_center: np.ndarray,
    lpa_radius: float,
    rpa_center: np.ndarray,
    rpa_radius: float,
    face_contour_rotated: np.ndarray | None = None,
    rect_half_width: float = 15.0,
    forehead_fallback: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Build a deletion mask from MediaPipe face contour + heuristic ears.

    Two-region mask in the eye-plane canonical frame (X=up, Y=anterior, Z=left):

    1. ``face_below`` -- everything below the nasion, anterior of the ears,
       strictly between the two ear spheres in Z.
    2. ``face_sides`` -- between the nasion (bottom) and a per-vertex forehead
       upper bound (top), same Z/Y constraints, EXCLUDING a vertical strip
       around the nasion Z (``rect_half_width``) that is kept for registration.

    The per-vertex forehead upper bound is interpolated from the rotated
    MediaPipe face-oval arc above the nasion, using KDTree on Z. When no
    contour is supplied the upper bound falls back to
    ``nasion[0] + forehead_fallback``.

    Args:
        verts: Mesh vertices in eye-plane frame, shape (N, 3).
        nasion: Rotated nasion position, shape (3,).
        lpa_center: LPA heuristic center in eye-plane frame, shape (3,).
        lpa_radius: LPA uncertainty-sphere radius (mm).
        rpa_center: RPA heuristic center in eye-plane frame, shape (3,).
        rpa_radius: RPA uncertainty-sphere radius (mm).
        face_contour_rotated: MediaPipe face oval points already rotated into
            the eye-plane frame, shape (M, 3). If None or contains fewer than 3
            points above the nasion, uses the fallback.
        rect_half_width: Z half-width of the preserved nasion strip (mm).
        forehead_fallback: Upper-bound offset above nasion if no contour (mm).

    Returns:
        Tuple of (mask, info). ``mask`` is a boolean array of shape (N,).
        ``info`` has keys ``arc_source`` (str), ``forehead_x_range``
        (tuple of float), ``y_cut`` (float), ``z_band`` (tuple of float),
        ``counts`` (dict of per-region counts).
    """
    z_band = (verts[:, 2] > rpa_center[2] + rpa_radius) & (
        verts[:, 2] < lpa_center[2] - lpa_radius
    )
    y_cut = min(0.5 * (lpa_center[1] + rpa_center[1]), nasion[1])
    front_half = verts[:, 1] > y_cut

    if face_contour_rotated is not None:
        contour = np.asarray(face_contour_rotated)
        forehead_arc = contour[contour[:, 0] > nasion[0]]
    else:
        forehead_arc = np.empty((0, 3))

    if len(forehead_arc) >= 3:
        arc_tree = KDTree(forehead_arc[:, 2:3])
        _, arc_idx = arc_tree.query(verts[:, 2:3], k=1)
        forehead_x_per_vertex = forehead_arc[arc_idx, 0]
        arc_source = f"MediaPipe forehead arc ({len(forehead_arc)} pts)"
    else:
        forehead_x_per_vertex = np.full(len(verts), nasion[0] + forehead_fallback)
        arc_source = f"fallback (nasion + {forehead_fallback:.0f}mm)"

    face_below = (verts[:, 0] < nasion[0]) & z_band & front_half

    in_arc_band = (verts[:, 0] >= nasion[0]) & (verts[:, 0] < forehead_x_per_vertex)
    nasion_strip = (
        (np.abs(verts[:, 2] - nasion[2]) < rect_half_width)
        & in_arc_band
        & front_half
    )
    face_sides = in_arc_band & z_band & front_half & ~nasion_strip

    mask = face_below | face_sides

    info = {
        "arc_source": arc_source,
        "forehead_x_range": (
            float(forehead_x_per_vertex.min()),
            float(forehead_x_per_vertex.max()),
        ),
        "y_cut": float(y_cut),
        "z_band": (
            float(rpa_center[2] + rpa_radius),
            float(lpa_center[2] - lpa_radius),
        ),
        "counts": {
            "face_below": int(face_below.sum()),
            "face_sides": int(face_sides.sum()),
            "nasion_strip_kept": int(nasion_strip.sum()),
            "all": int(mask.sum()),
        },
    }
    return mask, info
