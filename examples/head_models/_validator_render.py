"""Yaw/pitch sweep + MediaPipe BlazeFace, shared by detectability validators.

Notebooks 72 (three-way detectability comparison) and 73 (MediaPipe
boxes diagnostic) both render the head mesh from a 21-view camera sweep
and run BlazeFace on every view. The renderer + detector are factored
here so the camera geometry and detector configuration stay in lock-step
across both notebooks; if either one drifts, the thesis numbers diverge
across tables.

Camera convention (CTF frame, +X anterior, +Y left, +Z up):

- yaw=0, pitch=0 is true frontal (camera on +X looking at the centroid).
- Camera distance: 700 mm, fixed. Zoom: 1.3.
- Window: 640x640 px. Background flat white. Mesh shaded flat grey by
  default; pass ``use_color=True`` to paint the mesh with its per-vertex
  RGB (used for the colour contact-sheet renders).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pyvista as pv

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vision

from cedalion.vtktutils import trimesh_to_vtk_polydata


# Camera sweep
YAWS = list(range(-90, 91, 30))   # 7 yaws
PITCHES = [-20, 0, 20]            # 3 pitches
WINDOW = (640, 640)
GREY = (200, 200, 200)
CAM_DISTANCE_MM = 700.0
ZOOM = 1.3

# BlazeFace model bundled with the validation suite.
DEFAULT_BLAZE_PATH = (
    pathlib.Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"
)


def _vertex_rgba(mesh):
    """Return (n,4) uint8 vertex colors if the mesh carries them, else None."""
    visual = mesh.visual
    if hasattr(visual, "to_color"):
        try:
            v = visual.to_color()
            vc = getattr(v, "vertex_colors", None)
            if vc is not None and len(vc):
                return np.asarray(vc, dtype=np.uint8)
        except Exception:
            pass
    vc = getattr(visual, "vertex_colors", None)
    if vc is not None and len(vc):
        return np.asarray(vc, dtype=np.uint8)
    return None


def _wrap_with_color(mesh):
    """Wrap a trimesh into a pyvista PolyData, attaching RGB if available."""
    poly = pv.wrap(trimesh_to_vtk_polydata(mesh))
    rgba = _vertex_rgba(mesh)
    if rgba is not None and len(rgba) == poly.n_points:
        poly["RGB"] = rgba[:, :3]
    return poly


def render_views(surface, out_dir, tag, *, use_color=False):
    """Render the mesh from the yaw/pitch sweep at fixed 700 mm distance.

    yaw=0 / pitch=0 is a true frontal view in the CTF frame: camera placed
    on the +X axis (anterior), looking toward the head centroid, +Z up.

    Args:
        surface: TrimeshSurface in the CTF frame.
        out_dir: pathlib.Path; created if missing. PNGs are written here.
        tag: filename prefix (e.g., ``"original"``, ``"delete"``, ``"noise"``).
        use_color: when True, paint with per-vertex RGB (contact-sheet
            renders that should match the textured hero PNGs); when False,
            paint flat grey (detector sweep, identical baseline across
            variants).

    Returns:
        list of ``(yaw_deg, pitch_deg, path)`` tuples in iteration order.
    """
    pv.OFF_SCREEN = True
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poly = (
        _wrap_with_color(surface.mesh)
        if use_color
        else pv.wrap(trimesh_to_vtk_polydata(surface.mesh))
    )
    has_color = use_color and "RGB" in poly.array_names
    focal = np.asarray(surface.mesh.vertices).mean(axis=0)

    files = []
    for yaw in YAWS:
        for pitch in PITCHES:
            yaw_r, pitch_r = np.deg2rad(yaw), np.deg2rad(pitch)
            cam_dir = np.array([
                np.cos(yaw_r) * np.cos(pitch_r),
                np.sin(yaw_r) * np.cos(pitch_r),
                np.sin(pitch_r),
            ])
            p = pv.Plotter(off_screen=True, window_size=WINDOW)
            if has_color:
                p.add_mesh(poly, scalars="RGB", rgb=True, smooth_shading=True)
            else:
                p.add_mesh(poly, color=[c / 255 for c in GREY], smooth_shading=True)
            p.set_background("white")
            p.enable_anti_aliasing("ssaa")
            p.camera.position = tuple(focal + CAM_DISTANCE_MM * cam_dir)
            p.camera.focal_point = tuple(focal)
            p.camera.up = (0.0, 0.0, 1.0)
            p.camera.zoom(ZOOM)
            fn = out_dir / f"{tag}_yaw{yaw:+04d}_pitch{pitch:+03d}.png"
            p.screenshot(str(fn))
            p.close()
            files.append((yaw, pitch, fn))
    return files


def mediapipe_face_detector(model_path=DEFAULT_BLAZE_PATH, min_detection_confidence=0.5):
    """Construct a MediaPipe Tasks FaceDetector configured for short-range BlazeFace.

    Use this once at notebook scope; passing the resulting detector to
    `mediapipe_detect` / `mediapipe_detect_with_boxes` keeps the model
    loaded across the sweep.
    """
    return mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_py.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_detection_confidence,
        )
    )


def mediapipe_detect(image_path, detector):
    """Run BlazeFace on one rendered view; return ``(n_hits, max_confidence)``."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0.0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_img)
    if not res.detections:
        return 0, 0.0
    confs = [d.categories[0].score for d in res.detections]
    return len(confs), float(max(confs))


def mediapipe_detect_with_boxes(image_path, detector):
    """Run BlazeFace and draw bounding boxes on the rendered view.

    Returns ``(annotated_bgr, boxes)`` where ``boxes`` is a list of
    ``(xmin, ymin, w, h, score)`` tuples; the annotated image is the
    original BGR with green boxes drawn on every detection (no boxes
    when the detector returns nothing).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    boxes = []
    annotated = img.copy()
    if result.detections:
        for det in result.detections:
            bbox = det.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            score = float(det.categories[0].score)
            boxes.append((x, y, w, h, score))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 6, y), (0, 255, 0), -1)
            cv2.putText(
                annotated, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
            )
    return annotated, boxes
