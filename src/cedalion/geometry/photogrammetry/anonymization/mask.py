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

    Reindexes ``mesh.visual.uv`` (if present) in lockstep with the vertex array.
    Trimesh's ``remove_unreferenced_vertices`` does not touch UV arrays, so a
    naive call leaves ``len(uv) != len(vertices)`` and any subsequent textured
    export crashes or corrupts.

    Args:
        surface: Input TrimeshSurface.
        mask: Boolean array of shape (n_vertices,). True = vertex to remove.

    Returns:
        New TrimeshSurface with masked vertices (and the faces touching them)
        removed. CRS, units, UVs, and texture image are preserved in sync.
    """
    import trimesh

    old_mesh = surface.mesh
    old_verts = np.asarray(old_mesh.vertices)
    old_faces = np.asarray(old_mesh.faces)
    n_old_verts = len(old_verts)

    kept_face_mask = ~mask[old_faces].any(axis=1)
    kept_faces = old_faces[kept_face_mask]
    kept_vidx = np.unique(kept_faces)

    reindex = -np.ones(n_old_verts, dtype=np.int64)
    reindex[kept_vidx] = np.arange(len(kept_vidx))

    new_mesh = trimesh.Trimesh(
        vertices=old_verts[kept_vidx],
        faces=reindex[kept_faces],
        process=False,
    )

    old_visual = old_mesh.visual
    uv = getattr(old_visual, "uv", None)
    if uv is not None and len(uv) == n_old_verts:
        new_mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(uv)[kept_vidx],
            image=getattr(old_visual, "image", None),
            material=getattr(old_visual, "material", None),
        )
    else:
        vcol = getattr(old_visual, "vertex_colors", None)
        if vcol is not None and len(vcol) == n_old_verts:
            new_mesh.visual.vertex_colors = np.asarray(vcol)[kept_vidx]

    return cdc.TrimeshSurface(
        mesh=new_mesh, crs=surface.crs, units=surface.units,
    )


def _bake_vertex_colors(mesh) -> np.ndarray:
    """Sample the texture image at each vertex's UV to produce per-vertex RGBA.

    OBJ UVs use the bottom-left origin convention; PIL images use top-left,
    so V is flipped before indexing. UVs outside ``[0, 1]`` are wrapped
    (modulo 1) to match typical OBJ sampler behavior.

    Args:
        mesh: A ``trimesh.Trimesh`` with ``mesh.visual`` as ``TextureVisuals``.

    Returns:
        ``(n_vertices, 4)`` uint8 RGBA array. Falls back to uniform mid-grey
        (with a warning) if the mesh has no usable texture.
    """
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    image = getattr(visual, "image", None)
    if image is None:
        mat = getattr(visual, "material", None)
        image = getattr(mat, "image", None) if mat is not None else None
    n_verts = len(mesh.vertices)

    fallback = np.full((n_verts, 4), 180, dtype=np.uint8)
    fallback[:, 3] = 255

    if uv is None or image is None or len(uv) != n_verts:
        logger.warning(
            "with_color=True requested but mesh has no usable texture; "
            "falling back to uniform grey vertex colors."
        )
        return fallback

    img = np.asarray(image.convert("RGBA"))
    h, w = img.shape[:2]

    u = np.mod(np.asarray(uv)[:, 0], 1.0)
    v = np.mod(np.asarray(uv)[:, 1], 1.0)
    px = np.clip((u * w).astype(np.int64), 0, w - 1)
    py = np.clip(((1.0 - v) * h).astype(np.int64), 0, h - 1)

    return img[py, px]


def save_anonymized_scan(
    surface: cdc.TrimeshSurface,
    out_path: str,
    with_color: bool = False,
) -> list[str]:
    """Export an anonymized photogrammetry surface to disk.

    Always writes only the ``.obj`` file -- no MTL, no JPG. The original
    texture raster is never written alongside the mesh, so the subject's face
    pixels cannot leak through a saved image.

    - ``with_color=False`` (default): strip the texture entirely. Smallest
      file, geometry only.
    - ``with_color=True``: sample the texture at each kept vertex's UV and
      bake the result as per-vertex colors inline in the ``.obj``. The JPG
      is consumed in-memory and discarded; no raster is written.

    Args:
        surface: Anonymized TrimeshSurface (typically output of
            ``delete_masked_vertices``).
        out_path: Destination path ending in ``.obj``.
        with_color: If True, bake per-vertex colors from the texture.

    Returns:
        List of absolute paths written.

    Raises:
        ValueError: If ``out_path`` does not end in ``.obj``.
    """
    import os
    import trimesh

    if not out_path.lower().endswith(".obj"):
        raise ValueError(f"out_path must end in .obj, got: {out_path}")

    vcolors = _bake_vertex_colors(surface.mesh) if with_color else None

    mesh_to_write = trimesh.Trimesh(
        vertices=np.asarray(surface.mesh.vertices),
        faces=np.asarray(surface.mesh.faces),
        process=False,
    )
    if vcolors is not None:
        mesh_to_write.visual.vertex_colors = vcolors

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()

    mesh_to_write.export(out_path)

    after = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    stem = os.path.splitext(os.path.basename(out_path))[0]
    written = sorted(
        os.path.join(out_dir, f)
        for f in (after - before) | {os.path.basename(out_path)}
        if os.path.splitext(f)[0] == stem or f == os.path.basename(out_path)
    )

    logger.info(
        f"Saved anonymized scan (with_color={with_color}): "
        f"{[os.path.basename(p) for p in written]}"
    )
    return written
