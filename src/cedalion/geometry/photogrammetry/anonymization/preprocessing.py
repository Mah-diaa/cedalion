"""Preprocessing: axis normalization and head isolation.

Gets a raw Einstar photogrammetry scan into a canonical frame suitable for
landmark detection and mask building:

- ``normalize_axes`` rotates around X so Y points anterior (preliminary,
  used before head isolation).
- ``isolate_head`` strips body/shoulders/chair and disconnected fragments.
- ``align_axes_from_landmarks`` maps the scan into the CTF frame
  (+X=anterior, +Y=left, +Z=up, origin at the LPA-RPA midpoint) once all 5
  landmarks are available.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

logger = logging.getLogger("cedalion")


def _copy_visual(src_mesh, dst_mesh, vertex_index=None) -> None:
    """Copy a trimesh ``visual`` onto ``dst_mesh``, optionally reindexing.

    Assigning a ``TextureVisuals`` directly across meshes (e.g. by passing
    ``visual=old.visual`` to a new ``Trimesh(...)``) often silently downgrades
    it to ``ColorVisuals`` because the visual holds a back-reference to its
    original mesh and re-checks ``len(uv) == len(mesh.vertices)`` against the
    wrong target. Rebuilding the visual explicitly avoids that.

    Args:
        src_mesh: Source trimesh.
        dst_mesh: Destination trimesh (mutated in place).
        vertex_index: Optional indices into the source vertex array. If given,
            UVs / vertex_colors are sliced by this index before being attached.
            Leave None when the vertex count is unchanged.
    """
    import trimesh

    src_visual = src_mesh.visual
    uv = getattr(src_visual, "uv", None)
    image = getattr(src_visual, "image", None)
    n_src = len(src_mesh.vertices)

    if uv is not None and len(uv) == n_src:
        uv_arr = np.asarray(uv)
        if vertex_index is not None:
            uv_arr = uv_arr[vertex_index]
        if image is None:
            mat = getattr(src_visual, "material", None)
            image = getattr(mat, "image", None) if mat is not None else None
        dst_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_arr,
            image=image,
        )
        return

    vcol = getattr(src_visual, "vertex_colors", None)
    if vcol is not None and len(vcol) == n_src:
        vcol_arr = np.asarray(vcol)
        if vertex_index is not None:
            vcol_arr = vcol_arr[vertex_index]
        dst_mesh.visual.vertex_colors = vcol_arr


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
        x_max = vertices[:, 0].max()
        x_min = vertices[:, 0].min()
        upper = vertices[vertices[:, 0] > x_min + 0.6 * (x_max - x_min)]
        centroid_yz = np.array([0.0, upper[:, 1].mean(), upper[:, 2].mean()])
        nasion_yz = np.array([0.0, nasion[1], nasion[2]])
        forward_direction = nasion_yz - centroid_yz
        fwd_norm = np.linalg.norm(forward_direction)
        if fwd_norm < 1e-6:
            logger.warning("Cannot compute forward direction -- returning unchanged")
            return surface, nasion.copy(), np.eye(3)
        forward_direction = forward_direction / fwd_norm

    angle = np.arctan2(forward_direction[2], forward_direction[1])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])

    rotated_verts = vertices @ R.T

    import trimesh
    new_mesh = trimesh.Trimesh(
        vertices=rotated_verts,
        faces=surface.mesh.faces,
        process=False,
    )
    _copy_visual(surface.mesh, new_mesh)
    rotated_surface = cdc.TrimeshSurface(new_mesh, crs=surface.crs, units=surface.units)

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
    that sit far outside the head. Any step that uses vertex extrema will
    be dragged off into empty space by those fragments. Stripping them
    up-front prevents that.

    Uses ``trimesh.graph.connected_component_labels`` on face adjacency
    rather than ``mesh.split``; split allocates one full ``Trimesh`` per
    component and can OOM on scans with thousands of tiny fragments.

    Einstar OBJs duplicate vertices along UV seams for texturing. The
    default ``trimesh.Trimesh.merge_vertices()`` does NOT merge across UV
    seams (it preserves texture), so face adjacency on the raw mesh
    over-fragments the head into thousands of islands. We use POSITION-only
    merging via ``unique_rows`` for connectivity analysis.

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

    kept_vidx = np.unique(mesh.faces[face_mask])
    mask = np.zeros(n_verts, dtype=bool)
    mask[kept_vidx] = True
    return mask


def isolate_head(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    radius: float = 220.0,
) -> tuple[cdc.TrimeshSurface, np.ndarray]:
    """Remove shoulders, body, and chair; keep only the head.

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

    x_max = vertices[:, 0].max()
    x_min = vertices[:, 0].min()
    upper = vertices[:, 0] > x_min + 0.6 * (x_max - x_min)
    center = vertices[upper].mean(axis=0)
    midpoint_x = (x_max + nasion[0]) / 2.0
    center[0] = min(center[0], midpoint_x)

    dist = np.linalg.norm(vertices - center, axis=1)
    head_mask = dist < radius

    head_mask = head_mask & _largest_component_mask(surface.mesh)

    if head_mask.sum() < 100 or head_mask.mean() > 0.95:
        logger.debug(
            f"Head isolation: sphere contains {head_mask.mean()*100:.0f}% "
            f"of vertices -- scan is already head-only"
        )
        return surface, head_mask

    face_mask = head_mask[faces].all(axis=1)
    head_faces = faces[face_mask]

    kept_verts = np.unique(head_faces)
    reindex = np.full(len(vertices), -1, dtype=int)
    reindex[kept_verts] = np.arange(len(kept_verts))
    new_faces = reindex[head_faces]
    new_verts = vertices[kept_verts]

    new_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )
    _copy_visual(surface.mesh, new_mesh, vertex_index=kept_verts)

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


def align_axes_from_landmarks(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPointCloud,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPointCloud, np.ndarray]:
    """Map mesh + landmarks into the CTF anatomical frame.

    CTF convention:

        +X = anterior (toward Nz)
        +Y = left (toward Lpa)
        +Z = up (toward Cz)
        origin = midpoint of Lpa and Rpa (interaural midpoint)

    The returned surface and landmarks carry ``crs="ctf"``.

    Args:
        surface: Axis-normalized TrimeshSurface (post ``normalize_axes`` and
            ``isolate_head``).
        landmarks: LabeledPointCloud with labels Nz, Iz, Cz, Lpa, Rpa
            (matching the surface frame).

    Returns:
        Tuple of (aligned_surface, aligned_landmarks, transform).
        ``transform`` is a 4x4 homogeneous affine that maps input-frame
        points into CTF; apply as ``hom = transform @ [x, y, z, 1]``.
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
    origin = 0.5 * (Lpa + Rpa)

    y_ax = Lpa - Rpa
    y_ax = y_ax / np.linalg.norm(y_ax)

    nz_dir = Nz - origin
    nz_dir = nz_dir - np.dot(nz_dir, y_ax) * y_ax
    x_ax = nz_dir / np.linalg.norm(nz_dir)

    z_ax = np.cross(x_ax, y_ax)

    if np.dot(Cz - origin, z_ax) < 0:
        z_ax = -z_ax
    if np.dot(Nz - origin, x_ax) < 0:
        x_ax = -x_ax
    y_ax = np.cross(z_ax, x_ax)

    R = np.vstack([x_ax, y_ax, z_ax])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ origin

    aligned_verts = (np.asarray(surface.mesh.vertices) - origin) @ R.T
    new_mesh = trimesh.Trimesh(
        vertices=aligned_verts,
        faces=surface.mesh.faces,
        process=False,
    )
    _copy_visual(surface.mesh, new_mesh)
    aligned_surface = cdc.TrimeshSurface(
        new_mesh, crs="ctf", units=surface.units,
    )

    aligned_lm = (lm - origin) @ R.T
    old_crs_dim = [d for d in landmarks.dims if d != "label"][0]
    aligned_landmarks = (
        landmarks.pint.dequantify()
        .copy(data=aligned_lm)
        .rename({old_crs_dim: "ctf"})
        .pint.quantify()
    )

    return aligned_surface, aligned_landmarks, M


def revert_to_einstar_frame(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPointCloud,
    R_normalize: np.ndarray,
    M_align: np.ndarray,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPointCloud]:
    """Map an aligned surface and landmarks back into the raw Einstar frame.

    Inverse of the ``normalize_axes`` then ``align_axes_from_landmarks``
    composition, so the returned mesh and landmarks carry ``crs="digitized"``
    and match the coordinates of the original ``read_einstar_obj`` output.
    Useful right before ``save_anonymized_scan`` when the downstream
    co-registration pipeline expects the saved ``.obj`` and ``.tsv`` in the
    native scanner frame (e.g. Elsa's tutorial step 5.2 workflow, whose
    example files carry ``crs=digitized``).

    Note that ``isolate_head`` is not invertible: the returned mesh is still
    head-only even though its coordinates are in the original digitized frame.

    Args:
        surface: TrimeshSurface in the CTF frame (post
            ``align_axes_from_landmarks``, optionally after masking).
        landmarks: LabeledPointCloud in the CTF frame.
        R_normalize: 3x3 rotation returned by ``normalize_axes``.
        M_align: 4x4 affine returned by ``align_axes_from_landmarks``.

    Returns:
        Tuple of (surface_digitized, landmarks_digitized). Both carry
        ``crs="digitized"`` and the mesh preserves UVs / vertex colors.
    """
    import trimesh

    M_inv = np.linalg.inv(M_align)
    R_inv = R_normalize.T

    ctf_verts = np.asarray(surface.mesh.vertices)
    norm_verts = ctf_verts @ M_inv[:3, :3].T + M_inv[:3, 3]
    raw_verts = norm_verts @ R_inv.T

    new_mesh = trimesh.Trimesh(
        vertices=raw_verts,
        faces=surface.mesh.faces,
        process=False,
    )
    _copy_visual(surface.mesh, new_mesh)
    raw_surface = cdc.TrimeshSurface(
        new_mesh, crs="digitized", units=surface.units,
    )

    lm_ctf = landmarks.pint.dequantify().values
    lm_norm = lm_ctf @ M_inv[:3, :3].T + M_inv[:3, 3]
    lm_raw = lm_norm @ R_inv.T

    old_crs_dim = [d for d in landmarks.dims if d != "label"][0]
    raw_landmarks = (
        landmarks.pint.dequantify()
        .copy(data=lm_raw)
        .rename({old_crs_dim: "digitized"})
        .pint.quantify()
    )

    return raw_surface, raw_landmarks
