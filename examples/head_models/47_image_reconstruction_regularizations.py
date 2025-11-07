#!/usr/bin/env python
# coding: utf-8

# choose between two datasets
DATASET = "fingertappingDOT" # high-density montage
#DATASET = "fingertapping"   # sparse montage

# choose a head model
HEAD_MODEL = "colin27"
# HEAD_MODEL = "icbm152"

# choose between the monte
FORWARD_MODEL = "MCX" # photon monte carlo
#FORWARD_MODEL = "NIRFASTER" # finite element method - NOTE, you must have NIRFASTer installed via runnning <$ bash install_nirfaster.sh CPU # or GPU> from a within your cedalion root directory.

# set this flag to False to actual compute the forward model results
PRECOMPUTED_FLUENCE = True

# set this flag to True to enable interactive 3D plots
INTERACTIVE_PLOTS = False

import time
t1 = time.time()

import os

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as p
import numpy as np
import xarray as xr
from IPython.display import Image

import cedalion
import cedalion.dataclasses as cdc
import cedalion.datasets
import cedalion.geometry.registration
import cedalion.geometry.segmentation

import cedalion.io
import cedalion.plots
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.vis.plot_sensitivity_matrix
from cedalion import units
#from cedalion.imagereco.solver import pseudo_inverse_stacked

import cedalion.dot


from cedalion.io.forward_model import FluenceFile, load_Adot

import cedalion.xrutils as xrutils
xrutils.unit_stripping_is_error()


from scalene import scalene_profiler

#from IPython.core import display
display = print

xr.set_options(display_expand_data=False);

temporary_directory = TemporaryDirectory()
tmp_dir_path = Path(temporary_directory.name)


if DATASET == "fingertappingDOT":
    rec = cedalion.datasets.get_fingertappingDOT()
elif DATASET == "fingertapping":
    rec = cedalion.datasets.get_fingertapping()
else:
    raise ValueError("unknown dataset")


geo3d_meas = rec.geo3d
display(geo3d_meas)


meas_list = rec._measurement_lists["amp"]
display(meas_list.head(5))


if DATASET == "fingertappingDOT":
   rec.stim.cd.rename_events( {
        "1": "Control", 
        "2": "FTapping/Left", 
        "3": "FTapping/Right",
        "4": "BallSqueezing/Left",
        "5": "BallSqueezing/Right"
    } )
elif DATASET == "fingertapping":
    rec.stim.cd.rename_events( {
        "1.0": "Control", 
        "2.0": "FTapping/Left", 
        "3.0": "FTapping/Right"
    } )

# count number of trials per trial_type
display(
    rec.stim.groupby("trial_type")[["onset"]]
    .count()
    .rename({"onset": "#trials"}, axis=1)
)


# ## Preprocessing

rec["od"] = cedalion.nirs.int2od(rec["amp"])
rec["od_tddr"] = motion_correct.tddr(rec["od"])
rec["od_wavelet"] = motion_correct.wavelet(rec["od_tddr"])

# bandpass filter the data
rec["od_freqfiltered"] = rec["od_wavelet"].cd.freq_filter(
    fmin=0.01, fmax=0.5, butter_order=4
)

# ## Calculate block averages in optical density

# segment data into epochs
epochs = rec["od_freqfiltered"].cd.to_epochs(
    rec.stim,  # stimulus dataframe
    ["FTapping/Left", "FTapping/Right"],  # select fingertapping events, discard others
    before=5 * units.s,  # seconds before stimulus
    after=30 * units.s,  # seconds after stimulus
)

# calculate baseline
baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")

# subtract baseline
epochs_blcorrected = epochs - baseline

# group trials by trial_type. For each group individually average the epoch dimension
blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")

# ## The TwoSurfaceHeadModel
head = cedalion.dot.get_standard_headmodel(HEAD_MODEL)
head_ras = head.apply_transform(head.t_ijk2ras)
display(head_ras.crs)
display(head_ras.brain)

geo3d_snapped_ijk = head.align_and_snap_to_scalp(geo3d_meas)
display(geo3d_snapped_ijk)


# ## Forward Model
fwm = cedalion.dot.ForwardModel(head, geo3d_snapped_ijk, meas_list)

if PRECOMPUTED_FLUENCE:
    if FORWARD_MODEL == "MCX":
        fluence_fname = cedalion.datasets.get_precomputed_fluence(DATASET, HEAD_MODEL)
    elif FORWARD_MODEL == "NIRFASTER":
        raise NotImplementedError(
            "Currently there are no precomputed NIRFASTER results available"
        )
else:
    fluence_fname = tmp_dir_path / "fluence.h5"

    if FORWARD_MODEL == "MCX":
        fwm.compute_fluence_mcx(fluence_fname)
    elif FORWARD_MODEL == "NIRFASTER":
        fwm.compute_fluence_nirfaster(fluence_fname)


if PRECOMPUTED_FLUENCE:
    Adot = cedalion.datasets.get_precomputed_sensitivity(DATASET, HEAD_MODEL)
else:
    sensitivity_fname = tmp_dir_path / "sensitivity.h5"
    fwm.compute_sensitivity(fluence_fname, sensitivity_fname)
    Adot = load_Adot(sensitivity_fname)


# load and display sensitivity matrix
display(Adot)


# ## Reconstruct the Image


fname = Path("./sbf.h5")

USE_SBF = False

if USE_SBF:
    if fname.exists():
        sbf = cedalion.dot.GaussianSpatialBasisFunctions.from_file(fname)
    else:
        sbf = cedalion.dot.GaussianSpatialBasisFunctions(
            head_ras, Adot, **cedalion.dot.SBF_GAUSSIANS_DENSE
        )
        sbf.to_file(fname)
else:
    sbf = None



recon = cedalion.dot.ImageRecon(
    Adot,
    recon_mode="mua",
    regularization_params=cedalion.dot.RegularizationParams(
        alpha_meas=0.001, alpha_spatial=0.1, apply_c_meas=False
    ),
    spatial_basis_functions=sbf
)


#import os
#os.environ["SCALENE_PROFILE_ALL"] = "true"
#os.environ["SCALENE_OUTFILE"] = "scalene_clouseau.html"
#os.environ["SCALENE_HTML"] = "true"
#os.environ["SCALENE_CPU_ONLY"] = "false"     # or true
#os.environ["SCALENE_MEMORY_ONLY"] = "false"
#os.environ["SCALENE_NO_BROWSER"] = "true"
#
#
#
#scalene_profiler.start()


#recon_result = recon.reconstruct(blockaverage)
tr1 = time.time()
recon_result = recon.reconstruct(rec["od_freqfiltered"].sel(time=slice(0, 60)))
tr2 = time.time()
#scalene_profiler.stop()

display(recon_result)


t2 = time.time()


print(f"total runtime: {t2-t1:.3f}s. reco time: {tr2-tr1:3f}s")