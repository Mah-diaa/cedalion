"""Preprocessing: axis normalization and head isolation.

Gets a raw Einstar photogrammetry scan into a canonical frame suitable for
landmark detection and mask building:

- ``normalize_axes`` rotates around X so Y points anterior.
- ``isolate_head`` strips body/shoulders/chair and disconnected fragments.
- ``align_axes_from_landmarks`` derives a full rotation from the 5 anatomical
  landmarks once they are available.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

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
        visual=surface.mesh.visual,
        process=False,
    )
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
    try:
        old_visual = surface.mesh.visual
        if hasattr(old_visual, "vertex_colors") and len(old_visual.vertex_colors) == len(vertices):
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
