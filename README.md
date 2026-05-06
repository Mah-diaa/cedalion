# Face Anonymization for Photogrammetry Scans

**Thesis:** "Development of an Automatic Face Removal Algorithm for Photogrammetry Scans for Data Protection" — M.Sc. thesis, TU Berlin / IBS Lab

This is a fork of [cedalion](https://github.com/ibs-lab/cedalion) extended with a geometric face anonymization module for Einstar photogrammetry scans acquired in fNIRS research. The module lives at:

```
src/cedalion/geometry/photogrammetry/anonymization/
```

## Installation

Requires the cedalion conda environment:

```bash
conda env create -f environment_dev.yml
conda activate cedalion
pip install -e .
```

## Usage

### Interactive single-scan workflow (canonical)

Open `examples/head_models/51_manual_5pt_anonymization.ipynb`. The notebook:
1. Loads an Einstar scan (`cedalion.io.read_einstar_obj`)
2. Picks the five 10-20 landmarks interactively (Nz, Iz, Cz, LPA, RPA)
3. Calls `anonymize_scan(surface, landmarks)`
4. Shows a before/after comparison
5. Saves the anonymized OBJ + `_landmarks.tsv` sidecar via `save_anonymized_scan`

## Module structure

```
src/cedalion/geometry/photogrammetry/anonymization/
├── __init__.py        public API — re-exports all functions listed below
├── pipeline.py        anonymize_scan (canonical entry point)
├── preprocessing.py   normalize_axes, isolate_head, align_axes_from_landmarks,
│                      revert_to_einstar_frame
├── mask.py            detect_cap_boundary, face_mask_from_landmarks,
│                      delete_masked_vertices, save_anonymized_scan
└── _utils.py          private helpers shared by preprocessing and mask
                       (_rebuild_mesh, _copy_visual, _reindex_faces,
                        _apply_affine, _transform_labeled_points,
                        _ear_midpoint, _upper_head_centroid,
                        _resolve_texture_image)

examples/head_models/
└── 51_manual_5pt_anonymization.ipynb   interactive workflow notebook

tests/
└── test_anonymization.py               26 unit tests
```

Pipeline steps inside `anonymize_scan`:

1. `normalize_axes` — rotate so +Y points anterior (handles arbitrary Einstar orientation)
2. `isolate_head` — remove body, shoulders, and disconnected fragments
3. `align_axes_from_landmarks` — map to CTF frame (+X anterior, +Y left, +Z up)
4. `detect_cap_boundary` — locate the front cap-edge height along Z
5. `face_mask_from_landmarks` — face region union ear spheres, clamped below the cap
6. Landmark preservation — 8 mm spheres around each landmark + midline nasion strip
7. `delete_masked_vertices` — drop triangles touching any masked vertex, UVs in sync
8. `revert_to_einstar_frame` — return to `crs="digitized"` for saving

## Tests

```bash
pytest tests/test_anonymization.py -v
```

26 tests covering all eight public functions and the end-to-end pipeline. No real scan data is required: all tests build synthetic geometry using `trimesh.creation.icosphere`, which is the same approach used throughout the cedalion test suite (see `test_geodesics.py`, `test_dataclasses_geometry.py`). Three pytest fixtures are shared across tests:

- `simple_sphere_surface` — unit icosphere as a minimal `TrimeshSurface` for geometry-only checks
- `head_like_surface` — slightly elongated icosphere (X scaled ×1.2) that mimics head proportions and produces a non-trivial face region after masking
- `axis_normalized_landmarks` — five `LabeledPoints` (Nz, Iz, Cz, LPA, RPA) placed on the sphere axes, matching the post-`normalize_axes` coordinate frame

## Branch layout

| Branch | Contents |
|--------|---------|
| `feature/face-anonymization` | **This branch** — thesis implementation (anonymization module, notebook 51, test suite) |
| `main` | Upstream cedalion base |
| `validation/face-anonymization` | Validation notebooks 64-73, batch CSV producers |
| `auxiliary/mediapipe-nasion` | Experimental automatic nasion detection (MediaPipe) |

---

*Upstream cedalion documentation below.*

---

# Cedalion - fNIRS analysis toolbox

A python-based framework for the data-driven analysis of multimodal fNIRS and DOT in naturalistic environments. Developed by the [Intelligent Biomedical Sensing (IBS) Lab](https://ibs-lab.com/) with and for the community.

<p align="center">
    <img src="docs/img/cedalion_frontpage.png" />
</p>


## Documentation

The [documentation](https://doc.ibs.tu-berlin.de/cedalion/doc/dev) contains
[installation instructions](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/getting_started/installation.html), an [API reference](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/api/modules.html) as
well as many [example notebooks](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples.html) that illustrate the functionality of the toolbox.



## Installation

Please refer to the [installation instructions](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/getting_started/installation.html) in the documentation for installing Cedalion
on you computer.

To test the [example notebooks](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples.html) without installing Cedalion locally, you can also [run the notebooks on Google Colab](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/getting_started/colab_setup.html).

## Versions
The `main` branch acts as a relase branch, i.e. the latest commit there is the most 
current release. Development happens in the `dev` branch. Please refer to the [CHANGELOG](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/CHANGELOG.html) for a release 
history and current differences between the `dev` and `main` branches.


## Forum

For discussions and help please visit the [Cedalion forum on openfnirs.org](https://openfnirs.org/community/cedalion/)


## How to cite Cedalion
A paper for the toolbox is currently in the making. If you use this toolbox for a publication in the meantime, please cite us using GitHub's  "Cite this repository" feature in the "About" section. If you want to contact us or learn more about the IBS-Lab please go to https://www.ibs-lab.com/


## License

Cedalion is licensed under the MIT license.