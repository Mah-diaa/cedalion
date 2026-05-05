"""Optode co-registration invariance validator.

Detects colour-tagged optode stickers on the original and anonymized cap
scans, matches them per colour by nearest neighbour, and reports both the
raw sticker-centre displacement and the scalp-projected optode
displacement (centre minus optode length along the surface normal).

Sticker colours follow the supervisor's HSV ranges:

- ``'O'`` -- yellow, legacy / notebook 41 default; kept for backwards
  compatibility with older scans.
- ``'G'`` -- green = Detectors.
- ``'M'`` -- magenta = Sources.

The tuple format expected by ``ColoredStickerProcessor.colors`` is
``(hue_min, hue_max, value_min, value_max)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cedalion.geometry.photogrammetry.processors import ColoredStickerProcessor


DEFAULT_COLORS = {
    "O": (0.11, 0.21, 0.70, 1.0),
    "G": (0.14, 0.30, 0.30, 1.0),
    "M": (0.85, 1.00, 0.40, 1.0),
}

OPTODE_LENGTH_MM = 22.6

FACE_SPURIOUS_RADIUS_MM = 10.0
"""Drop original-side detections without an anonymized counterpart within
this distance. These are face-region false positives that disappear
after anonymization; keeping them would break the 1:1 cap-sticker
correspondence."""


def detect_stickers_by_group(surface, colors=DEFAULT_COLORS):
    """Run the sticker processor and split detections by colour group.

    Args:
        surface: TrimeshSurface to process.
        colors: Mapping from group label to ``(h_min, h_max, v_min, v_max)``.

    Returns:
        Dict ``{group: (centres_mm, normals)}`` with both arrays of shape
        ``(n_in_group, 3)``. Groups with zero detections are omitted.
    """
    proc = ColoredStickerProcessor(colors=colors)
    centres, normals = proc.process(surface, details=False)
    if len(centres) == 0:
        return {}
    centres_np = centres.pint.dequantify().values
    normals_np = normals.values
    groups = np.asarray(centres.group.values)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[str(g)] = (centres_np[mask], normals_np[mask])
    return out


def match_one_to_one(c_orig, c_anon, max_distance_mm=FACE_SPURIOUS_RADIUS_MM):
    """Greedy strict-1:1 sticker pairing capped at ``max_distance_mm``.

    Sorts all pairwise distances ascending and greedily picks the
    smallest-distance pair whose endpoints are still free. Stops when no
    remaining pair is within ``max_distance_mm``. Anything beyond that
    radius is treated as a non-cap detection (typically face-region
    false positives that the orig detector picks up but the anonymized
    detector cannot, since the face is gone).

    Result is strict 1:1: each original and each anonymized sticker
    appears in at most one pair, and every reported pair is within
    ``max_distance_mm`` of itself.
    """
    if len(c_orig) == 0 or len(c_anon) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    cost = np.linalg.norm(c_orig[:, None, :] - c_anon[None, :, :], axis=-1)
    n_a = cost.shape[1]
    order = np.argsort(cost, axis=None)
    used_o, used_a = set(), set()
    pairs = []
    for k in order:
        if cost.flat[k] > max_distance_mm:
            break
        i, j = divmod(int(k), n_a)
        if i in used_o or j in used_a:
            continue
        pairs.append((i, j))
        used_o.add(i)
        used_a.add(j)
    if not pairs:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    i_o, i_a = map(np.array, zip(*pairs))
    return i_o, i_a


def compute_subject_deviations(
    surface_orig,
    surface_anon,
    colors=DEFAULT_COLORS,
    optode_length_mm=OPTODE_LENGTH_MM,
    face_spurious_radius_mm=FACE_SPURIOUS_RADIUS_MM,
):
    """Match stickers per colour and return per-sticker deviations.

    Args:
        surface_orig: Original mesh.
        surface_anon: Anonymized mesh.
        colors: HSV-range mapping passed to ``ColoredStickerProcessor``.
        optode_length_mm: Distance from sticker centre to scalp along the
            inward normal, used for the scalp-projected displacement.

    Returns:
        Tuple ``(rows, counts)``.

        ``rows`` is a list of dicts, one per 1:1 matched pair, with keys
        ``group``, ``sticker_idx``, ``sticker_dev_mm``, ``scalp_dev_mm``.

        ``counts`` is ``{group: n_matched}`` for every colour group with
        at least one matched pair.
    """
    by_group_orig = detect_stickers_by_group(surface_orig, colors)
    by_group_anon = detect_stickers_by_group(surface_anon, colors)

    counts = {}
    rows = []
    for g in sorted(set(by_group_orig) & set(by_group_anon)):
        c_o, n_o = by_group_orig[g]
        c_a, n_a = by_group_anon[g]
        i_o, i_a = match_one_to_one(c_o, c_a, max_distance_mm=face_spurious_radius_mm)
        if len(i_a) == 0:
            continue
        co_m = c_o[i_o]
        ca_m = c_a[i_a]
        no_m = n_o[i_o]
        na_m = n_a[i_a]
        sticker_dev = np.linalg.norm(co_m - ca_m, axis=1)
        scalp_o = co_m - optode_length_mm * no_m
        scalp_a = ca_m - optode_length_mm * na_m
        scalp_dev = np.linalg.norm(scalp_o - scalp_a, axis=1)
        counts[g] = len(i_a)
        for k in range(len(i_a)):
            rows.append({
                "group": g,
                "sticker_idx": int(i_a[k]),
                "sticker_dev_mm": float(sticker_dev[k]),
                "scalp_dev_mm": float(scalp_dev[k]),
            })
    return rows, counts


def summarize_per_group(df_per_sticker):
    """Aggregate per-sticker deviations into per-group statistics."""
    return (
        df_per_sticker.groupby("group")
        .agg(
            n=("sticker_dev_mm", "size"),
            sticker_mean_mm=("sticker_dev_mm", "mean"),
            sticker_median_mm=("sticker_dev_mm", "median"),
            sticker_max_mm=("sticker_dev_mm", "max"),
            scalp_mean_mm=("scalp_dev_mm", "mean"),
            scalp_median_mm=("scalp_dev_mm", "median"),
            scalp_max_mm=("scalp_dev_mm", "max"),
        )
        .reset_index()
    )
