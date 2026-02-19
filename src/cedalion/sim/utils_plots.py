import random

import numpy as np
import pandas as pd
import pyvista as pv
import scipy.stats as stats
import xarray as xr
from typing import Annotated
import cedalion.dataclasses as cdc
import cedalion.dataclasses.geometry as cdg
import cedalion.dot.forward_model as cfm
import cedalion.vis
import cedalion.typing as cdt
from cedalion import Quantity, units
import cedalion.models.glm as glm
from cedalion.models.glm.basis_functions import TemporalBasisFunction
import cedalion.vis.blocks as vbx
from scipy.fft import fftfreq
from scipy.signal import spectrogram, welch, detrend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne


def plot_fnirs(fnirs_ts, channels=None, stim=None, fmt=None, figsize=(12, 2)):
    """Plot fNIRS data"""

    fnirs_ts = fnirs_ts.pint.dequantify()

    # Check dimensionality and order dimensions
    if len(fnirs_ts.dims) != 3:
        raise ValueError("fNIRS data must have 3 dimensions: channel-like, wavelength/chromo, time")
    if 'time' not in fnirs_ts.dims:
        raise ValueError("fNIRS data must have 'time' dimension")
    if 'wavelength' in fnirs_ts.dims:
        ty1, ty2 = fnirs_ts.wavelength.values
        colors = ['blue', 'orange'] if ty1 < ty2 else ['orange', 'blue']
        fnirs_ts = fnirs_ts.transpose(..., 'wavelength', 'time')
    elif 'chromo' in fnirs_ts.dims:
        ty1, ty2 = fnirs_ts.chromo.values
        colors = ['red', 'blue'] if ty1 == 'HbO' else ['blue', 'red']
        fnirs_ts = fnirs_ts.transpose(..., 'chromo', 'time')
    else:
        raise ValueError("fNIRS data must have either 'wavelength' or 'chromo' dimension")

    channel_dim = fnirs_ts.dims[0]
    if channels is None:
        channels = fnirs_ts[channel_dim].values.tolist()
    else:
        for ch in channels:
            if ch not in fnirs_ts[channel_dim].values:
                raise ValueError(f"Channel {ch} not found in fNIRS data")
    Nc = len(channels)
    figsize = (figsize[0], figsize[1] * Nc)
    fig, axs = plt.subplots(Nc, 1, figsize=figsize, sharex=True)
    axs = [axs] if Nc == 1 else axs  # Ensure axs is always a list
    for i, ch in enumerate(channels):
        ch_data = fnirs_ts.sel({channel_dim: ch}).values
        for j, types in enumerate([ty1, ty2]):
            axs[i].plot(fnirs_ts.time, 
                        ch_data[j], 
                        color=colors[j],
                        label=types if i == 0 else "")
        
        if stim is not None:
            # for _, row in stim.iterrows():
            #     axs[i].axvspan(row['onset'], row['onset'] + row['duration'], color='gray', alpha=0.2, label=row['trial_type'] if i == 0 else "")
            vbx.plot_stim_markers(axs[i], stim, fmt=fmt, y=1)

        axs[i].set_title(f'fNIRS, Channel: {ch}')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid()
        if i == 0:
            axs[i].legend()
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_eeg(eeg_raw, duration=5, channels=30, 
             fmax=None, plot_psd=False, plot_montage=False, scalings='auto'):
    """Plot EEG data, PSD and montage."""
    if isinstance(channels, int):
        eeg_raw.plot(n_channels=channels, duration=duration, scalings=scalings)
    elif isinstance(channels, list):
        eeg_raw.plot(picks=channels, duration=duration, scalings=scalings)
    if plot_psd:
        fmax = fmax if fmax is not None else eeg_raw.info['sfreq'] / 2
        eeg_raw.compute_psd(fmax=fmax).plot()
    if plot_montage:
        eeg_raw.plot_sensors(show_names=True)


def plot_activation(
    spatial_img: xr.DataArray,
    head,
    title: str = "",
    montage = None,
    show_labels=False
):
    """Plots a spatial activation pattern on the brain surface.

    Args:
        spatial_img (xr.DataArray): Activation values for each vertex.
        brain (TrimeshSurface): Brain Surface with brain mesh.
        seed (int): Seed vertex for the activation pattern.
        title (str): Title for the plot.
        log_scale (bool): Whether to map activations on a logarithmic scale.

    Returns:
        None

    Initial Contributors:
        - Thomas Fischer | t.fischer.1@campus.tu-berlin.de | 2024

    """

    brain = head.brain
    scalp = head.scalp

    # Store original colors to restore later
    old_colors = brain.mesh.visual.vertex_colors.copy()

    seed = spatial_img.argmax()
    vertices = brain.mesh.vertices
    center_brain = np.mean(vertices, axis=0)
    colors_blob = get_colors(spatial_img, brain.mesh.visual.vertex_colors)
    brain.mesh.visual.vertex_colors = colors_blob
    plt_pv = pv.Plotter()
    vbx.plot_surface(plt_pv, brain)
    vbx.plot_surface(plt_pv, scalp, opacity=0.1)
    if montage is not None:
        vbx.plot_labeled_points(plt_pv, montage, show_labels=show_labels)
    plt_pv.camera.position = (vertices[seed] - center_brain) * 7 + center_brain
    plt_pv.camera.focal_point = vertices[seed]
    plt_pv.camera.up = [0, 0, 1]
    plt_pv.add_text(title, position="upper_edge", font_size=20)
    plt_pv.show()

    # Restore original colors
    brain.mesh.visual.vertex_colors = old_colors


def get_colors(
    activations: xr.DataArray,
    vertex_colors: np.array,
    log_scale: bool = False,
    max_scale: float = None,
):
    """Maps activations to colors for visualization.

    Args:
        activations (xr.DataArray): Activation values for each vertex.
        vertex_colors (np.array): Vertex color array of the brain mesh.
        log_scale (bool): Whether to map activations on a logarithmic scale.
        max_scale (float): Maximum value to scale the activations.

    Returns:
        np.array: New vertex color array with same shape as `vertex_colors`.
    """

    if not isinstance(activations, np.ndarray):
        activations = activations.pint.dequantify()
    # linear scale:
    if max_scale is None:
        max_scale = activations.max()
    activations = (activations / max_scale) * 255
    # map on a logarithmic scale to the range [0, 255]
    if log_scale:
        activations = np.log(activations + 1)
        activations = (activations / max_scale) * 255
    activations = activations.astype(np.uint8)
    colors = np.zeros((vertex_colors.shape), dtype=np.uint8)
    colors[:, 3] = 255
    colors[:, 0] = 255
    activations = 255 - activations
    colors[:, 1] = activations
    colors[:, 2] = activations
    colors[colors < 0] = 0
    colors[colors > 255] = 255

    return colors


def plot_parcellated_surface(plt_pv, brain_surface, parcel_file):
    """Plot brain surface with parcels colored differently."""

    parcels_df = cedalion.io.read_parcellations(parcel_file)
    all_parcels = list(set(brain_surface.vertices.parcel.values))
   
    # map parcel labels to colors
    parcel_colors = {}
    for i, parcel in enumerate(all_parcels):
        parcel_info = parcels_df[parcels_df['Label'] == parcel]
        if not parcel_info.empty:
            color = parcel_info.iloc[0]['Color']
            parcel_colors[parcel] = (color[0], color[1], color[2], 1)
        else:
            parcel_colors[parcel] = (0.8, 0.8, 0.8, 1)  # default gray

    # Assign colors to vertices
    b = cdc.VTKSurface.from_trimeshsurface(brain_surface)
    b = pv.wrap(b.mesh)
    b["parcels"] = np.asarray([
        parcel_colors.get(parcel, (0.8, 0.8, 0.8, 1))
        for parcel in brain_surface.vertices.parcel.values
    ])


    plt_pv.add_mesh(
        b,
        scalars="parcels",
        rgb=True,
        smooth_shading=False
    )

# def restrict_to_subset(x, time_lim=None, feature_name=None, features=None):
#     """
#     Take a time series and restrict to time and feature dimension
#     """
    
#     x_sub = x.copy()

#     # Define subset from x by restricting time and/or features
#     ## Times
#     if time_lim is not None:
#         x_sub = x_sub.sel(time=slice(*time_lim))

#     ## Features
#     if features is not None:
#         x_sub = x_sub.sel({feature_name: features})

#     return x_sub


# def plot_multiple_timeseries(x_list,
#                      colors=None,
#                      labels=None, 
#                      time_lim=None):
    
#     """
#     Plot generic time series with dimensions feature_name x time.
#     It can plot a subset of features and a subset of time.
#     """

#     N_series = len(x_list)
#     if N_series == 1:
#         x_list = [x_list]
    
#     # Define subset from x by restricting time
#     if time_lim is not None:
#         ti, tf = time_lim
#         x_sub = [x.loc[(x.time >= ti)*(x.time <= tf)] for x in x_list if time_lim is not None]
#     else:
#         x_sub = x_list
        
#     # Define ylim from min max values
#     ymin = [x.min() for x in x_sub]
#     ymax = [x.max() for x in x_sub]
#     ylim = (np.min(ymin), np.max(ymax))

#     fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
    
#     for x, c, l in zip(x_sub, colors, labels):
#         ax.plot(x.time, x, c, label=l)        
    
#     ax.set_ylim(*ylim)
#     ax.grid()
#     ax.legend()
#     plt.tight_layout()
#     plt.xlabel('Time (s)')


# def scalp_plot_spatial_pattern(A, figsize=(5, 5)):
#     """
#     Make a scalp plot from spattial patterns A of shape (N_channels)
#     """

#     # Create the MNE info object
#     info = mne.create_info(ch_names=A.channel.data.tolist(), sfreq=100, ch_types=['eeg'] * 32)
#     raw = mne.io.RawArray(A.data.reshape(-1, 1), info); 
#     # Set the electrode layout
#     montage = mne.channels.make_standard_montage('easycap-M1')
#     raw.set_montage(montage)

#     # Normalize
#     A /= np.abs(A).max()
#     min_max = (A.min(), A.max())
#     # Plot
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     im, cn = mne.viz.plot_topomap(A.data, raw.info, axes=ax, show=False, vlim=min_max); 
#     cax = fig.colorbar(im, ax=ax)
#     cax.set_label(r'Activity (A.U)')
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()



# def scalp_plots(x, rate):

#     # Create the MNE info object
#     info = mne.create_info(ch_names=x.channel.data.tolist(), sfreq=rate, ch_types=['eeg'] * 32)
#     raw = mne.io.RawArray(x, info)
#     # Set the electrode layout
#     montage = mne.channels.make_standard_montage('easycap-M1')
#     raw.set_montage(montage)

#     # Define equally spaced time points (in seconds)
#     min_max = (x.min(), x.max())
#     num_plots = 10  # Number of scalp plots to generate
#     T = x.time.data[-1] # Total duration in seconds
#     time_points = np.linspace(0, T, num_plots)

#     # Create scalp plots
#     fig, axes = plt.subplots(num_plots//4 + 1, 3, figsize=(12, 12))  # Adjust grid size based on `num_plots`
#     axes = axes.flatten()  # Flatten axes for easy indexing

#     for i, time_point in enumerate(time_points):
#         sample_idx = int(time_point * rate)
        
#         if sample_idx == x.shape[1]:
#             continue
        
#         # Compute the average of all previous points
#         if sample_idx > 0:
#             data_avg = np.mean(raw._data[:, :sample_idx], axis=1)
#         else:
#             data_avg = raw._data[:, 0]  # If first time point, use the first sample
        
#         data_avg = raw._data[:, sample_idx]
        
#         # Create scalp plot on the respective subplot
#         im, cn = mne.viz.plot_topomap(data_avg, raw.info, axes=axes[i], show=False, vlim=min_max)
#         cax = fig.colorbar(im, ax=axes[i])
#         cax.set_label(r'Activity')
#         axes[i].set_title(f'Time: {time_point:.2f}s')

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()


# def plot_head_and_montage(head, montage):
#     plt_pv = pv.Plotter()
#     vbx.plot_surface(plt_pv, head.brain, color="#d3a6a1")
#     vbx.plot_surface(plt_pv, head.scalp, opacity=.1)
#     vbx.plot_labeled_points(plt_pv, montage, show_labels=True)
#     plt_pv.show()


# def plot_all(sim):

#     # Define variables
#     sx = sim.s_temporal_eeg
#     sy = sim.s_temporal_fnirs

#     # Visualize random temporal source
#     fig, ax = plt.subplots(2*sim.Ns, 1, figsize=(15, 6*sim.Ns), sharex=True)

#     ax = [ax] if type(ax) is not np.ndarray else ax

#     for i in range(0, 2*sim.Ns, 2):
        
#         # Timeseries 
#         ax[i].plot(sim.time_sim, sy[i//2], 'r', label='fNIRS')
#         ax[i].plot(sim.time_sim, sx[i//2], 'b', label='EEG', alpha=0.5)
#         ax[i].grid()
#         for v in sim.stim.onset.values:
#             ax[i].axvline(v, color='k')
#         for v in sim.stim.offset.values:
#             ax[i].axvline(v, color='k')
#         ax[i].legend()
#         ax[i].set_title(sim.source_locations[i//2])
        
#         # Spectrogram
#         freqs, t_spec, spec = spectrogram(sx[i//2], fs=sim.rate, nperseg=500, noverlap=250)
#         i += 1
#         mask = freqs <= 20
#         freqs = freqs[mask]
#         spec_db = 10 * np.log10(spec[mask])
#         img = ax[i].pcolormesh(t_spec, freqs, spec_db, shading='gouraud')
#         ax[i].set_ylabel('Frequency [Hz]')
#         ax[i].set_xlabel('Time [s]')
#     #     # Add a colorbar to the right of each subplot
#     #     cbar = plt.colorbar(img, ax=ax[i])
#     #     cbar.set_label('Power (dB)')
#         for v in sim.stim.onset.values:
#             ax[i].axvline(v, color='k')
#         for v in sim.stim.offset.values:
#             ax[i].axvline(v, color='k')
#         ax[i].legend()
#         ax[i].set_title(sim.source_locations[i//2])

#     plt.xlabel('Time (s)')
#     plt.tight_layout()
#     plt.show()

#     # Visualize OD in channel
#     min_max = (sim.y_od.sel(wavelength=760).min(), sim.y_od.sel(wavelength=850).max())
#     # Channels to plot
#     channels_to_plot = sim.y_od.channel[['S4' in c for c in sim.y_od.channel.data]].data[4:7]
#     Nc = len(channels_to_plot)

#     fig, ax = plt.subplots(Nc, 1, figsize=(15, 3*Nc), sharex=True)

#     for i, c in enumerate(channels_to_plot):
#         ax[i].plot(sim.y_od.time, sim.y_od.sel(wavelength=760, channel=c), label='760')
#         ax[i].plot(sim.y_od.time, sim.y_od.sel(wavelength=850, channel=c), 'r', label='850')
#         ax[i].plot(sim.y_ba_compatible.time, sim.y_ba_compatible.sel(wavelength=760, channel=c), 'b--')
#         ax[i].plot(sim.y_ba_compatible.time, sim.y_ba_compatible.sel(wavelength=850, channel=c), 'r--', label='background')
#         for v in sim.stim.onset.values:
#             ax[i].axvline(v, color='k')
        
#         ax[i].legend()
#         ax[i].set_ylabel('Optical Density')
#     #     ax[i].set_ylim(*min_max)
#         ax[i].grid()
#         ax[i].set_title(f'Channel {c}')
        
#     plt.tight_layout()
#     plt.xlabel('Time (s)')
#     plt.show()

#     # Visualize EEG channels
#     # min_max = (sim.x.min(), sim.x.max())
#     # Channels to plot
#     channels_to_plot = sim.eeg_channels[4:8]
#     Nc = len(channels_to_plot)

#     fig, ax = plt.subplots(Nc, 1, figsize=(15, 3*Nc), sharex=True)

#     for i, c in enumerate(channels_to_plot):
#         ax[i].plot(sim.time_sim, sim.x.sel(channel=c))
#         ax[i].plot(sim.x_ba_compatible.time, sim.x_ba_compatible.sel(channel=c), label='background')
#         ax[i].set_ylabel('V')
#     #     ax[i].set_ylim(*min_max)
#         ax[i].legend()
#         ax[i].grid()
#         ax[i].set_title(f'Channel {c}')
#         ax[i].set_xlim(0, sim.T_sim)
#         for v in sim.stim.onset.values:
#             ax[i].axvline(v, color='k')
#         for v in sim.stim.offset.values:
#             ax[i].axvline(v, color='k')
#     plt.tight_layout()
#     plt.xlabel('Time (s)')
#     plt.show()




# def plot_time_series(x, 
#                      x_background=None, 
#                      colors=None, 
#                      time_lim=None,  
#                      features=None, 
#                      feature_name='channel', 
#                      xlim=None, 
#                      stim=None,
#                      background_color=None):
#     """
#     Plot generic time series with dimensions feature_name x time.
#     It can plot a subset of features and a subset of time. 
#     x can contain several time series, each with their corresponding label and color
#     """

#     # Define subset from x by restricting time and/or features
#     x_sub = restrict_to_subset(x, time_lim, feature_name, features)
#     if x_background is not None:
#         x_sub_ba = restrict_to_subset(x_background, time_lim, feature_name, features)
#     time = x_sub.time.data
#     features = x_sub[feature_name].data
    
#     # Identify extra dimension (assume only 1 extra)
#     extra_dims = [d for d in x_sub.dims if d not in [feature_name, 'time']]
#     if len(extra_dims) == 0:
#         extra_dims = 0
#     elif len(extra_dims) == 1:
#         extra_dims = extra_dims[0]
#     else:
#         raise ValueError('x does not has 2 nor 3 dimensions...')

#     if stim is not None:
#         trial_types = stim.source.unique()
#         trial_colors = ['green', 'orange', 'magenta', 'blue'][:len(trial_types)]
#         stim = stim[stim.onset < time[-1]]

#     if colors is None:
#         colors = ['blue', 'red']

#     ylim = (x_sub.min(), x_sub.max())

#     N = len(features)
#     fig, ax = plt.subplots(N, 1, figsize=(10, 3*N), sharex=True)

#     # Catch flat axis
#     ax = [ax] if N==1 else ax

#     for i, f in enumerate(features):
#         if extra_dims:
#             for d, c in zip(x_sub[extra_dims].data, colors):
#                 ax[i].plot(time, x_sub.sel({feature_name: f, extra_dims: d}), color=c, label=f'{d}')
#                 if x_background is not None:
#                     ax[i].plot(time, x_sub_ba.sel({feature_name: f, extra_dims: d}), color=c, linestyle='--', label=f'{d} background')
#             ax[i].legend()
#         else:
#             if x_background is not None:
#                 color_ba = background_color  if background_color else 'b'
#                 ax[i].plot(time, x_sub_ba.sel({feature_name: f}), color=color_ba, linestyle='--', label='background')
#                 ax[i].legend()
#             ax[i].plot(time, x_sub.sel({feature_name: f}), 'b')
            
        
#         # Plot onsets if specified
#         if stim is not None:
#             for s, tc in zip(trial_types, trial_colors):
#                 stim_tmp = stim[stim.source==s]
#                 for o in stim_tmp['onset'].values:
#                     ax[i].vlines(o, ymin=-1, ymax=1, colors=tc, alpha=0.3)

        
#         ax[i].set_ylim(*ylim)
#         ax[i].grid()
#         ax[i].set_title(f'{feature_name}: {f}')
#         if xlim is not None:
#             ax[i].set_xlim(*xlim)
        


#     plt.tight_layout()
#     plt.xlabel('Time (s)')
#     plt.show()


# def plot_sci_psp_quality(amp, montage, perc_time_clean, sci_x_psp_mask):
#     """
#     Plot scalp plot and windowed time series of fnirs data with a SCI and PSP quality assessment.
#     """

#     # plot the percentage of clean time per channel
#     f,ax = plt.subplots(1,1,figsize=(8,8))

#     vbx.scalp_plot(
#         amp,
#         montage,
#         perc_time_clean,
#         ax,
#         cmap="RdYlGn",
#         vmin=0,
#         vmax=1,
#         title=None,
#         cb_label="Percentage of clean time",
#         channel_lw=2
#     )
#     f.tight_layout()

#     # we can also plot this as a binary heatmap
#     f,ax = plt.subplots(1,1,figsize=(10,20))

#     sci_binary_cmap = LinearSegmentedColormap.from_list("sci_binary_cmap", list(zip([0,0.5,0.5,1],["#DC3220","#DC3220","#0C7BDC","#0C7BDC"])))
#     m = ax.pcolormesh(sci_x_psp_mask.time, np.arange(len(sci_x_psp_mask.channel)), sci_x_psp_mask, shading="nearest", cmap=sci_binary_cmap)
#     cb = plt.colorbar(m, ax=ax)
#     plt.tight_layout()
#     ax.yaxis.set_ticks(np.arange(len(sci_x_psp_mask.channel)))
#     ax.yaxis.set_ticklabels(sci_x_psp_mask.channel.values);
#     cb.set_label("PSP > 0.1 and SCI > 0.75")
#     ax.set_xlabel("time / s");