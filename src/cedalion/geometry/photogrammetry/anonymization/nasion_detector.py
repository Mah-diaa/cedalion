"""Automatic nasion detection from 3D mesh geometry.

Detects the nasion (Nz) from an Einstar scan without user interaction.
Only assumes X (index 0) = Up (gravity-based). The forward direction
in the Y-Z horizontal plane is determined automatically using MediaPipe
Face Landmarker: render the mesh from multiple viewpoints, detect the
face in 2D, back-project landmarks to 3D.

Returns None when automatic detection fails (e.g. EEG cap false
positives). The caller should fall back to manual nasion selection.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging
import pathlib

import numpy as np
import xarray as xr
from scipy.spatial import KDTree, cKDTree

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

from .preprocessing import normalize_axes

logger = logging.getLogger("cedalion")

_MODEL_DIR = pathlib.Path(__file__).parent
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _filter_contour_outliers(
    contour_3d: np.ndarray,
    max_sigma: float = 2.5,
) -> np.ndarray:
    """Remove outlier points from the face contour.

    Rejects points whose distance from the contour centroid exceeds
    max_sigma standard deviations. This catches "floating bits" from
    seam cracks where VTK picks a disconnected mesh fragment.

    Args:
        contour_3d: Face contour points, shape (N, 3).
        max_sigma: Rejection threshold in standard deviations.

    Returns:
        Filtered contour points, shape (M, 3) where M <= N.
    """
    centroid = contour_3d.mean(axis=0)
    dists = np.linalg.norm(contour_3d - centroid, axis=1)
    std_dist = dists.std()
    if std_dist < 1e-6:
        return contour_3d
    keep = dists < dists.mean() + max_sigma * std_dist
    return contour_3d[keep]


def detect_nasion_auto(
    surface: cdc.TrimeshSurface,
) -> tuple[np.ndarray, dict] | None:
    """Automatically detect the nasion using MediaPipe face detection.

    Renders the mesh from multiple viewpoints, runs MediaPipe Face
    Landmarker to find the face, then refines the nasion via
    midsagittal profile analysis.

    Returns None when detection fails -- the caller should fall back
    to manual nasion selection (see ``pick_nasion`` in ``ui.py``).

    Args:
        surface: TrimeshSurface from photogrammetry scan.

    Returns:
        Tuple of (nasion_position, metadata) or None if detection fails.
        nasion_position is a numpy array of shape (3,) in mm.
        metadata is a dict with keys: method, confidence, nose_tip,
        forward_direction, face_contour_3d, eyes. ``eyes`` is a tuple
        (r_eye_3d, l_eye_3d) -- eye-corner midpoints from the same
        validated view as the nasion, or None on profile-fallback paths.
    """
    vertices = surface.mesh.vertices

    # --- Step 1: Isolate head (remove chair/body) ---
    head_verts, _ = _isolate_head_vertices(vertices)

    # --- Step 2: YZ centroid of upper head (needed for profile analysis) ---
    x_min = head_verts[:, 0].min()
    x_max = head_verts[:, 0].max()
    upper = head_verts[head_verts[:, 0] > x_min + 0.6 * (x_max - x_min)]
    yz_centroid = np.array([0.0, upper[:, 1].mean(), upper[:, 2].mean()])

    # --- Step 3: Try MediaPipe face detection ---
    ml_result = _find_nose_tip_mediapipe(surface.mesh, head_verts)

    if ml_result is None:
        logger.info("Automatic nasion detection failed — no face found")
        return None

    (
        nose_tip_ml,
        forward_dir,
        nasion_proxy,
        frontal_reliable,
        face_contour,
        r_eye_3d,
        l_eye_3d,
    ) = ml_result
    logger.debug(
        f"MediaPipe face found, "
        f"fwd_angle={np.degrees(np.arctan2(forward_dir[2], forward_dir[1])):.0f}deg, "
        f"frontal_reliable={frontal_reliable}, "
        f"face_contour={'yes' if face_contour is not None else 'no'}"
    )

    # --- Primary: use validated bundle nasion directly ---
    if frontal_reliable:
        nasion = _snap_to_original(nasion_proxy, vertices)
        logger.debug(f"Auto nasion (unified): {nasion}")
        return nasion, {
            "method": "mediapipe+unified",
            "confidence": 0.90,
            "nose_tip": nose_tip_ml.copy(),
            "forward_direction": forward_dir.copy(),
            "face_contour_3d": face_contour,
            "eyes": (r_eye_3d.copy(), l_eye_3d.copy()),
        }

    # --- Fallback: geometric profile analysis ---
    # The unified sweep only returns frontal_reliable=True, so this branch
    # is effectively dead code. Kept in case a future sweep variant returns
    # an unvalidated bundle.
    result = _try_direction(
        head_verts, vertices, forward_dir, yz_centroid,
    )
    if result is not None:
        result["method"] = "mediapipe+profile"
        if result["dip"] > 0:
            result["confidence"] = 0.85
        else:
            result["confidence"] = 0.7
        logger.debug(
            f"Auto nasion (ML+profile): {result['nasion']}, "
            f"confidence={result['confidence']:.2f}, dip={result['dip']:.2f}"
        )
        return result["nasion"], {
            "method": result["method"],
            "confidence": result["confidence"],
            "nose_tip": result["nose_tip"],
            "forward_direction": result["forward_direction"],
            "face_contour_3d": face_contour,
            "eyes": (r_eye_3d.copy(), l_eye_3d.copy()),
        }

    logger.info("Profile analysis failed after MediaPipe detection")
    return None


# ---------------------------------------------------------------------------
# MediaPipe-based detection
# ---------------------------------------------------------------------------

def _find_nose_tip_mediapipe(
    mesh_trimesh,
    head_verts: np.ndarray,
    cam_distance: float = 400.0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray | None,
    np.ndarray, np.ndarray
] | None:
    """Unified MediaPipe sweep: nasion, nose tip, eye corners, face oval
    all extracted from the single best view.

    Renders the mesh from up to 72 viewpoints (24 azimuth x 3 pitch). For
    each view where MediaPipe detects a face, back-projects nasion (lm 168),
    nose tip (lm 1), both eye-corner pairs (33/133 + 263/362), and the face
    oval (36 contour indices) using the same vtkCellPicker. Jointly validates
    the bundle (all points on-mesh, X ordering nasion>nose, eyes roughly at
    nasion height), ranks surviving views by |yaw|, returns the bundle from
    the view with smallest |yaw|. Early-exits when a view with |yaw|<5 deg
    validates.

    Args:
        mesh_trimesh: trimesh.Trimesh mesh object (with texture).
        head_verts: Head-isolated vertices for centroid computation.
        cam_distance: Camera distance from head centroid in mm.

    Returns:
        Tuple (nose_tip_3d, forward_direction, nasion_3d, frontal_reliable,
        face_contour, r_eye_3d, l_eye_3d) or None if no view validates.
        frontal_reliable is True when the bundle came from joint validation
        (i.e. the new code path). r_eye_3d / l_eye_3d are the eye-corner
        midpoints (outer+inner)/2 for each eye.
    """
    try:
        import pyvista as pv
        import mediapipe as mp
        import vtk
    except ImportError as e:
        raise ImportError(
            "Automatic nasion detection requires mediapipe. "
            "Install with: pip install cedalion[anonymization]"
        ) from e

    if not _MODEL_PATH.exists():
        logger.info(f"MediaPipe model not found at {_MODEL_PATH}")
        return None

    head_centroid = head_verts.mean(axis=0)
    x_min_h = head_verts[:, 0].min()
    x_max_h = head_verts[:, 0].max()

    # Build pyvista mesh with texture colors
    faces_pv = np.column_stack([
        np.full(len(mesh_trimesh.faces), 3), mesh_trimesh.faces
    ]).ravel()
    pv_mesh = pv.PolyData(mesh_trimesh.vertices, faces_pv)

    try:
        colors = mesh_trimesh.visual.to_color().vertex_colors[:, :3]
        pv_mesh["colors"] = colors
        has_colors = True
    except Exception:
        has_colors = False

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    if has_colors:
        plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)
    else:
        plotter.add_mesh(pv_mesh, color="bisque")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_facial_transformation_matrixes=True,
    )

    # MediaPipe FaceLandmarker indices
    _NASION = 168
    _NOSE_TIP = 1
    _R_OUT, _R_IN = 33, 133
    _L_OUT, _L_IN = 263, 362
    _FACE_OVAL_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109,
    ]

    result = None
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as lm:
            # === Unified sweep: all landmarks from the same best view ===
            # Azimuth sweeps around the X=up axis in the Y-Z plane (step 15deg).
            # Pitch tilts the camera up/down around the tangent axis, so
            # subjects whose face is tilted relative to gravity can still be
            # captured. For each view where MediaPipe detects a face, we
            # back-project nasion, nose tip, both eye-corner pairs, and the
            # face oval -- all from the SAME rendering. The bundle is jointly
            # validated and ranked by |yaw|. Guarantees nasion + eyes come
            # from the same view, so they can't disagree.
            bundles: list[dict] = []
            best_so_far_yaw = 180.0

            for theta_deg in range(0, 360, 15):
                theta = np.radians(theta_deg)
                # Horizontal direction from centroid in Y-Z plane
                horiz = np.array([0.0, np.cos(theta), np.sin(theta)])

                for pitch_deg in (-20, 0, 20):
                    pitch = np.radians(pitch_deg)
                    # Tilt the camera by rotating around the tangent axis
                    # (horizontal, perpendicular to horiz in Y-Z plane).
                    cam_dir = (
                        np.cos(pitch) * horiz
                        + np.sin(pitch) * np.array([1.0, 0.0, 0.0])
                    )
                    cam_pos = head_centroid + cam_distance * cam_dir

                    plotter.camera_position = [
                        cam_pos.tolist(),
                        head_centroid.tolist(),
                        [1, 0, 0],
                    ]
                    plotter.render()
                    img = plotter.screenshot(return_img=True)
                    H, W = img.shape[:2]

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(img),
                    )
                    detection = lm.detect(mp_image)

                    if not detection.face_landmarks:
                        continue

                    # --- Yaw ranking (skip obviously-profile views) ---
                    yaw_abs = 180.0
                    if detection.facial_transformation_matrixes:
                        from scipy.spatial.transform import Rotation
                        mat = detection.facial_transformation_matrixes[0]
                        rot = Rotation.from_matrix(np.array(mat)[:3, :3])
                        yaw = rot.as_euler("yxz", degrees=True)[0]
                        yaw_abs = abs(yaw)
                        if yaw_abs > 60:
                            continue

                    face = detection.face_landmarks[0]

                    picker = vtk.vtkCellPicker()
                    picker.SetTolerance(0.01)

                    def _bp(idx: int) -> np.ndarray:
                        lm_pt = face[idx]
                        picker.Pick(
                            float(int(lm_pt.x * W)),
                            float(H - 1 - int(lm_pt.y * H)),
                            0, plotter.renderer,
                        )
                        return np.array(picker.GetPickPosition())

                    nasion_3d = _bp(_NASION)
                    nose_3d = _bp(_NOSE_TIP)
                    r_out = _bp(_R_OUT)
                    r_in = _bp(_R_IN)
                    l_out = _bp(_L_OUT)
                    l_in = _bp(_L_IN)

                    # --- Joint validation: all six key points on-mesh ---
                    key_pts = [nasion_3d, nose_3d, r_out, r_in, l_out, l_in]
                    if any(np.allclose(p, 0) for p in key_pts):
                        continue

                    # X-ordering: nasion above nose tip (X = gravity-up)
                    if nasion_3d[0] <= nose_3d[0]:
                        continue

                    # Reject nose on top of head
                    nose_rel = (nose_3d[0] - x_min_h) / (x_max_h - x_min_h)
                    if nose_rel > 0.85:
                        continue

                    r_eye_3d = 0.5 * (r_out + r_in)
                    l_eye_3d = 0.5 * (l_out + l_in)

                    # Eyes must be above nose tip
                    if (r_eye_3d[0] <= nose_3d[0]
                            or l_eye_3d[0] <= nose_3d[0]):
                        continue

                    # Eye midline should be close to nasion in height (within 30mm)
                    eye_x_mid = 0.5 * (r_eye_3d[0] + l_eye_3d[0])
                    if abs(eye_x_mid - nasion_3d[0]) > 30.0:
                        continue

                    # Eye corners not too close to each other
                    eye_sep = float(np.linalg.norm(r_eye_3d - l_eye_3d))
                    if eye_sep < 30.0 or eye_sep > 120.0:
                        continue

                    # Forward direction: nose tip -> centroid in YZ plane
                    fwd = nose_3d - head_centroid
                    fwd[0] = 0.0
                    fwd_norm = np.linalg.norm(fwd)
                    if fwd_norm < 1e-6:
                        continue
                    fwd = fwd / fwd_norm

                    # --- Face oval from the SAME view ---
                    contour_pts = []
                    for idx in _FACE_OVAL_INDICES:
                        pt = _bp(idx)
                        if not np.allclose(pt, 0):
                            contour_pts.append(pt.copy())
                    if len(contour_pts) >= 20:
                        raw_contour = np.array(contour_pts)
                        face_contour = _filter_contour_outliers(raw_contour)
                        if len(face_contour) < 20:
                            face_contour = raw_contour
                    else:
                        face_contour = None

                    bundles.append({
                        "yaw_abs": yaw_abs,
                        "nasion_3d": nasion_3d.copy(),
                        "nose_3d": nose_3d.copy(),
                        "r_eye_3d": r_eye_3d.copy(),
                        "l_eye_3d": l_eye_3d.copy(),
                        "fwd": fwd.copy(),
                        "face_contour": face_contour,
                        "theta_deg": theta_deg,
                        "pitch_deg": pitch_deg,
                    })

                    # Early-exit: if we have a near-frontal validated bundle,
                    # stop sweeping -- further views won't improve on this.
                    if yaw_abs < 5.0:
                        logger.debug(
                            f"Unified sweep: early exit at theta={theta_deg}, "
                            f"pitch={pitch_deg} (|yaw|={yaw_abs:.1f})"
                        )
                        best_so_far_yaw = yaw_abs
                        break

                    if yaw_abs < best_so_far_yaw:
                        best_so_far_yaw = yaw_abs

                else:
                    continue
                break

            if not bundles:
                logger.debug("Unified sweep: no view passed joint validation")
                result = None
            else:
                bundles.sort(key=lambda b: b["yaw_abs"])
                best = bundles[0]
                logger.debug(
                    f"Unified sweep: {len(bundles)} valid views, best "
                    f"theta={best['theta_deg']}deg pitch={best['pitch_deg']}deg "
                    f"|yaw|={best['yaw_abs']:.1f}"
                )
                result = (
                    best["nose_3d"].copy(),
                    best["fwd"].copy(),
                    best["nasion_3d"].copy(),
                    True,  # frontal_reliable: bundle jointly validated
                    best["face_contour"],
                    best["r_eye_3d"].copy(),
                    best["l_eye_3d"].copy(),
                )
    except Exception as e:
        logger.warning(f"MediaPipe detection error: {e}")
    finally:
        plotter.close()

    return result


# ---------------------------------------------------------------------------
# Geometric detection (fallback)
# ---------------------------------------------------------------------------

def _try_direction(
    head_verts: np.ndarray,
    original_verts: np.ndarray,
    forward_dir: np.ndarray,
    yz_centroid: np.ndarray,
    nose_tip_override: np.ndarray | None = None,
) -> dict | None:
    """Try a candidate forward direction and return nasion result with dip score.

    Args:
        head_verts: Head-isolated vertices.
        original_verts: Original mesh vertices for snapping.
        forward_dir: Candidate forward unit vector (X=0).
        yz_centroid: YZ centroid of upper head (X=0).
        nose_tip_override: If provided, use this as the nose tip instead of
            detecting it geometrically. Useful when ML provides a nose tip.

    Returns:
        Dict with nasion, confidence, dip score, etc., or None on failure.
    """
    lateral_dir = np.cross(np.array([1.0, 0.0, 0.0]), forward_dir)
    lateral_dir = lateral_dir / np.linalg.norm(lateral_dir)

    if nose_tip_override is not None:
        nose_tip = nose_tip_override.copy()
    else:
        nose_tip, _ = _find_nose_tip(head_verts, forward_dir)

    lat_dist = np.abs(np.dot(head_verts - nose_tip, lateral_dir))
    midline = lat_dist < 10.0
    above = head_verts[:, 0] > nose_tip[0]
    fwd_proj = np.dot(head_verts, forward_dir)
    centroid_fwd = np.dot(yz_centroid, forward_dir)
    anterior = fwd_proj > centroid_fwd

    mask = midline & above & anterior
    candidates = np.where(mask)[0]

    if len(candidates) < 5:
        mask = midline & above
        candidates = np.where(mask)[0]

    if len(candidates) < 5:
        return None

    # Bin by X height, take median forward-projection per bin
    cand_x = head_verts[candidates, 0]
    cand_fwd = fwd_proj[candidates]

    x_min, x_max = cand_x.min(), cand_x.max()
    n_bins = max(10, int((x_max - x_min) / 1.0))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.clip(np.digitize(cand_x, bin_edges) - 1, 0, n_bins - 1)

    bin_fwd = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = bin_indices == b
        if in_bin.sum() > 0:
            bin_fwd[b] = np.median(cand_fwd[in_bin])

    valid = ~np.isnan(bin_fwd)
    bin_centers_valid = bin_centers[valid]
    bin_fwd_valid = bin_fwd[valid]

    if len(bin_fwd_valid) < 3:
        return None

    # Smooth profile to suppress mesh noise
    window = min(7, len(bin_fwd_valid))
    if window >= 3:
        kernel = np.ones(window) / window
        padded = np.pad(bin_fwd_valid, (window // 2, window // 2), mode='edge')
        bin_fwd_smoothed = np.convolve(padded, kernel, mode='valid')
    else:
        bin_fwd_smoothed = bin_fwd_valid

    # Search in anatomically plausible range: nose_tip to nose_tip + 40mm
    nose_tip_x = nose_tip[0]
    search_mask = (
        (bin_centers_valid >= nose_tip_x)
        & (bin_centers_valid <= nose_tip_x + 40)
    )
    search_idx = np.where(search_mask)[0]

    if len(search_idx) >= 3:
        nasion_bin = _find_deepest_minimum(
            bin_fwd_smoothed[search_idx], min_dip=0.5
        )
        if nasion_bin is not None:
            nasion_bin = search_idx[nasion_bin]
        else:
            nasion_bin = search_idx[np.argmin(bin_fwd_smoothed[search_idx])]
    else:
        nasion_bin = _find_deepest_minimum(bin_fwd_smoothed, min_dip=0.5)
        if nasion_bin is None:
            half = max(1, len(bin_fwd_smoothed) // 2)
            nasion_bin = np.argmin(bin_fwd_smoothed[:half])

    nasion_x = bin_centers_valid[nasion_bin]
    nasion_fwd_val = bin_fwd_valid[nasion_bin]

    # Find best candidate vertex near this height
    bin_half = (bin_centers_valid[1] - bin_centers_valid[0]) if len(
        bin_centers_valid) > 1 else 2.0
    height_match = np.abs(head_verts[candidates, 0] - nasion_x) < bin_half * 2
    if height_match.sum() > 0:
        height_cands = candidates[height_match]
        fwd_dist = np.abs(fwd_proj[height_cands] - nasion_fwd_val)
        best_cand = height_cands[np.argmin(fwd_dist)]
        nasion_approx = head_verts[best_cand]
    else:
        nasion_approx = np.array([nasion_x, nose_tip[1], nose_tip[2]])
    nasion = _snap_to_original(nasion_approx, original_verts)

    # Dip depth (using smoothed profile)
    if 0 < nasion_bin < len(bin_fwd_smoothed) - 1:
        dip = (min(bin_fwd_smoothed[nasion_bin - 1], bin_fwd_smoothed[nasion_bin + 1])
               - bin_fwd_smoothed[nasion_bin])
    else:
        dip = 0.0

    confidence = float(min(1.0, dip / 5.0))

    return {
        "nasion": nasion,
        "nose_tip": nose_tip.copy(),
        "forward_direction": forward_dir.copy(),
        "confidence": confidence,
        "dip": dip,
        "method": "profile",
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _isolate_head_vertices(
    vertices: np.ndarray,
    max_head_height: float = 280.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isolate head vertices by cropping to within max_head_height of the top.

    Note: This is a fast height-based heuristic for nasion detection only.
    The public ``isolate_head()`` in face_detector.py uses sphere-based
    filtering that preserves mesh connectivity for downstream processing.

    Args:
        vertices: Mesh vertices of shape (N, 3).
        max_head_height: Maximum expected head height in mm.

    Returns:
        Tuple of (head_vertices, mask) where mask is boolean array.
    """
    x_max = vertices[:, 0].max()
    mask = vertices[:, 0] > x_max - max_head_height

    if mask.sum() < 100:
        logger.warning("Head isolation: too few vertices, using all")
        mask = np.ones(len(vertices), dtype=bool)

    logger.debug(f"Head isolation: {len(vertices)} -> {mask.sum()} vertices")
    return vertices[mask], mask


def _find_nose_tip(
    vertices: np.ndarray,
    forward_direction: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Find nose tip as the most-forward vertex in the upper head.

    Args:
        vertices: Head-isolated vertices of shape (N, 3).
        forward_direction: Unit vector pointing forward in Y-Z plane.

    Returns:
        Tuple of (nose_tip_position, vertex_index).
    """
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    x_threshold = x_min + 0.6 * (x_max - x_min)
    upper_mask = vertices[:, 0] > x_threshold

    if upper_mask.sum() == 0:
        upper_mask = np.ones(len(vertices), dtype=bool)

    fwd_proj = np.dot(vertices, forward_direction)
    fwd_proj[~upper_mask] = -np.inf
    idx = np.argmax(fwd_proj)

    return vertices[idx].copy(), idx


def _find_deepest_minimum(values: np.ndarray, min_dip: float = 0.5):
    """Find the deepest local minimum that dips at least min_dip below neighbors.

    Args:
        values: 1D array of values.
        min_dip: Minimum dip depth to be considered significant.

    Returns:
        Index of the deepest significant local minimum, or None.
    """
    best_idx = None
    best_dip = 0.0
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            dip = min(values[i - 1], values[i + 1]) - values[i]
            if dip >= min_dip and dip > best_dip:
                best_dip = dip
                best_idx = i
    return best_idx


def _snap_to_original(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Snap a point to the nearest original mesh vertex.

    Args:
        point: Target position of shape (3,).
        vertices: Original mesh vertices of shape (N, 3).

    Returns:
        Nearest vertex position as numpy array of shape (3,).
    """
    tree = KDTree(vertices)
    _, idx = tree.query(point)
    return vertices[idx].copy()


def derive_landmarks_from_nasion(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
) -> cdt.LabeledPoints:
    """Derive Iz, Cz, LPA, RPA from a known nasion + surface geometry.

    Internally normalizes the mesh axes around X using ``nasion`` (so Y
    points anterior, Z left), picks the four remaining landmarks from
    extrema in that frame, and rotates everything back to the input
    surface's frame so the returned ``LabeledPoints`` is a drop-in for
    the manual picker output and can be fed straight into
    ``anonymize_scan``.

    Args:
        surface: TrimeshSurface in raw Einstar coordinates (X=up, Y/Z
            arbitrary). The output ``crs`` is reused.
        nasion: Nasion position of shape (3,) in mm, in the same frame
            as ``surface``.

    Returns:
        ``LabeledPoints`` with labels ``Nz, Iz, Cz, LPA, RPA``,
        positions in the input surface's frame, units of mm.
    """
    nasion = np.asarray(nasion, dtype=float)

    surface_norm, nasion_norm, R = normalize_axes(surface, nasion)
    vertices_norm = np.asarray(surface_norm.mesh.vertices)

    centroid = vertices_norm.mean(axis=0)
    head_height = vertices_norm[:, 0].max() - vertices_norm[:, 0].min()
    band = 0.20 * head_height

    # Cz: highest vertex (max X) near midline.
    lateral_mask = (
        (np.abs(vertices_norm[:, 1] - centroid[1]) < band)
        & (np.abs(vertices_norm[:, 2] - centroid[2]) < band)
    )
    if lateral_mask.sum() == 0:
        lateral_mask = np.ones(len(vertices_norm), dtype=bool)
    cz_idx = np.where(lateral_mask)[0][np.argmax(vertices_norm[lateral_mask, 0])]
    cz = vertices_norm[cz_idx]

    # Iz: most posterior (min Y) at nasion height on the midsagittal plane.
    iz_height_mask = np.abs(vertices_norm[:, 0] - nasion_norm[0]) < 20.0
    iz_midline_mask = np.abs(vertices_norm[:, 2] - nasion_norm[2]) < 25.0
    iz_mask = iz_height_mask & iz_midline_mask
    if iz_mask.sum() == 0:
        iz_mask = np.abs(vertices_norm[:, 0] - nasion_norm[0]) < 40.0
    iz_cands = np.where(iz_mask)[0]
    iz = vertices_norm[iz_cands[np.argmin(vertices_norm[iz_cands, 1])]]

    # LPA/RPA: geometric targets in the ear band, snapped to surface.
    midline_z = np.mean([nasion_norm[2], iz[2], cz[2]])
    clean_band = (
        (vertices_norm[:, 0] > nasion_norm[0] + 10.0)
        & (vertices_norm[:, 0] < cz[0] - 20.0)
    )
    if clean_band.sum() > 100:
        clean_z = vertices_norm[clean_band, 2]
        half_width = (np.percentile(clean_z, 97) - np.percentile(clean_z, 3)) / 2.0
    else:
        half_width = 75.0

    lpa_target = np.array([nasion_norm[0], cz[1], midline_z + half_width])
    rpa_target = np.array([nasion_norm[0], cz[1], midline_z - half_width])

    ear_region = (
        (np.abs(vertices_norm[:, 0] - nasion_norm[0]) < 30.0)
        & (np.abs(vertices_norm[:, 1] - cz[1]) < 40.0)
    )

    def _snap(target):
        if ear_region.sum() > 10:
            idxs = np.where(ear_region)[0]
            tree = cKDTree(vertices_norm[idxs])
            _, local = tree.query(target)
            return vertices_norm[idxs[local]]
        tree = cKDTree(vertices_norm)
        _, idx = tree.query(target)
        return vertices_norm[idx]

    lpa = _snap(lpa_target)
    rpa = _snap(rpa_target)

    # Rotate the four derived landmarks back into the surface's input frame.
    # ``normalize_axes`` produces ``rotated_nasion = R @ nasion``, so the
    # inverse mapping is ``p_raw = R.T @ p_norm``.
    R_inv = R.T
    nz_raw = nasion
    iz_raw = R_inv @ iz
    cz_raw = R_inv @ cz
    lpa_raw = R_inv @ lpa
    rpa_raw = R_inv @ rpa

    labels = ["Nz", "Iz", "Cz", "LPA", "RPA"]
    coords = np.vstack([nz_raw, iz_raw, cz_raw, lpa_raw, rpa_raw])
    landmarks = xr.DataArray(
        coords,
        dims=["label", surface.crs],
        coords={
            "label": labels,
            "type": ("label", [cdc.PointType.LANDMARK] * 5),
            "group": ("label", ["L"] * 5),
        },
    ).pint.quantify("mm")

    logger.debug(
        f"Derived 4 landmarks from Nz: Iz={iz_raw}, Cz={cz_raw}, "
        f"LPA={lpa_raw}, RPA={rpa_raw}"
    )

    return landmarks
