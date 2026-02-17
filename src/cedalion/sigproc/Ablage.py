
# Lager für den Summenplot, damit ich ihn aus der Datei "bvp_wav_ana_v12"
# heruasnehemn kann.
# Für die Ausführung des Summenplots siehe
# "BVPA_Algo_Wieland_Impl_Plots_Comments_Jan18_FSv1_b.pdf"


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

from cedalion.sigproc.frequency import sampling_rate
from cedalion.dataclasses.bvp_container import BVP_Container

def cmap_parula():

    parula_data = np.array([
        [0.2081, 0.1663, 0.5292],
        [0.2116, 0.1898, 0.5777],
        [0.2123, 0.2138, 0.6270],
        [0.2081, 0.2386, 0.6771],
        [0.1959, 0.2645, 0.7279],
        [0.1707, 0.2919, 0.7792],
        [0.1253, 0.3242, 0.8303],
        [0.0591, 0.3598, 0.8683],
        [0.0117, 0.3875, 0.8820],
        [0.0057, 0.4086, 0.8828],
        [0.0165, 0.4266, 0.8786],
        [0.0329, 0.4430, 0.8720],
        [0.0498, 0.4586, 0.8641],
        [0.0629, 0.4737, 0.8554],
        [0.0723, 0.4887, 0.8467],
        [0.0779, 0.5040, 0.8384],
        [0.0793, 0.5200, 0.8312],
        [0.0749, 0.5375, 0.8263],
        [0.0641, 0.5570, 0.8240],
        [0.0488, 0.5772, 0.8228],
        [0.0343, 0.5966, 0.8199],
        [0.0265, 0.6137, 0.8135],
        [0.0239, 0.6287, 0.8038],
        [0.0231, 0.6418, 0.7913],
        [0.0228, 0.6535, 0.7768],
        [0.0267, 0.6642, 0.7607],
        [0.0384, 0.6743, 0.7436],
        [0.0590, 0.6838, 0.7254],
        [0.0843, 0.6928, 0.7062],
        [0.1133, 0.7015, 0.6859],
        [0.1453, 0.7098, 0.6646],
        [0.1801, 0.7177, 0.6424],
        [0.2178, 0.7250, 0.6193],
        [0.2586, 0.7317, 0.5954],
        [0.3022, 0.7376, 0.5712],
        [0.3482, 0.7424, 0.5473],
        [0.3953, 0.7459, 0.5244],
        [0.4420, 0.7481, 0.5033],
        [0.4871, 0.7491, 0.4840],
        [0.5300, 0.7491, 0.4661],
        [0.5709, 0.7485, 0.4494],
        [0.6099, 0.7473, 0.4337],
        [0.6473, 0.7456, 0.4188],
        [0.6834, 0.7435, 0.4044],
        [0.7184, 0.7411, 0.3905],
        [0.7525, 0.7384, 0.3768],
        [0.7858, 0.7356, 0.3633],
        [0.8185, 0.7327, 0.3498],
        [0.8507, 0.7299, 0.3360],
        [0.8824, 0.7274, 0.3217],
        [0.9139, 0.7258, 0.3063],
        [0.9450, 0.7261, 0.2886],
        [0.9739, 0.7314, 0.2666],
        [0.9938, 0.7455, 0.2403],
        [0.9990, 0.7653, 0.2164],
        [0.9955, 0.7861, 0.1967],
        [0.9880, 0.8066, 0.1794],
        [0.9789, 0.8271, 0.1633],
        [0.9697, 0.8481, 0.1475],
        [0.9626, 0.8705, 0.1309],
        [0.9589, 0.8949, 0.1132],
        [0.9598, 0.9218, 0.0948],
        [0.9661, 0.9514, 0.0755],
        [0.9763, 0.9831, 0.0538],
    ])

    parula = LinearSegmentedColormap.from_list("parula", parula_data)

    return parula

def plot_coherence_bvpa_pr(bvp_cont: BVP_Container, ch: str,
                           coherence_thresh=0.9,
                           arrow_step_time=30,
                           arrow_step_period: int=4) -> None:
    """Plots time series and wavelet coherence between BVPA and PR.

    The mean phase is calculated for the whole frequency range but plotted only for
    the frequencies where arrwos are plotted.

    Args:
        bvp_cont: BVP Container which includes the blood volume pulse time series
            created by the function "extract_bvp" and the two storages.
        ch: string that specifies the channel which sould be plotted.
        coherence_thresh: threshold above which the phase-arrows should be plotted.
        arrow_step_time: steps between lines of arrow-grid in the direction of time
            in seconds.
        arrow_step_period: steps between lines of arrow-grid in the direction of
            frequency. Higher values lead to lower arrow density.

    Example:
        plot_concts_bvpats_pr(rec, "S1D15")
    """

    WCT = bvp_cont.wav_storage_details[ch]["wavelet_coherence"]
    aWCT = bvp_cont.wav_storage_details[ch]["phase"]
    coi = bvp_cont.wav_storage_details[ch]["cone_of_interest"]
    freq = bvp_cont.wav_storage_details[ch]["frequency"]
    significance = bvp_cont.wav_storage_details[ch]["significance"]
    time = bvp_cont.wav_storage_details[ch]["wc_time"] / 60
    S12 = bvp_cont.wav_storage_details[ch]["cross_wavelet_transform"]
    S1 = bvp_cont.wav_storage_details[ch]["cwt_signal1"]
    S2 = bvp_cont.wav_storage_details[ch]["cwt_signal2"]
    n = time.size

    fs_qty = sampling_rate(bvp_cont['bvpa_ts'])
    fs = float(fs_qty.to('Hz').magnitude)
    arrow_step_time = int(arrow_step_time * fs)

    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    # ----- PLOT -----
    fig = plt.figure(figsize=(13.5, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[7, 1],
                  height_ratios=[1, 1], wspace=0.08, hspace=0.27)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.08)

    color_bvpa = [0.259, 0.478, 0.729]
    color_pr  = [0.959, 0.278, 0.329]

    # ----- Comparison BVPA and PR
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(bvp_cont['bvpa_ts'].time / 60,
                bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_smooth"),
                color=color_bvpa, linewidth=0.5, label='BVPA')
    ax_top.set_ylabel('BVPA [AU]')
    ax_top.set_xlabel('Time [min]')
    ax_top.set_zorder(1)

    ax_top_pr = ax_top.twinx()
    ax_top_pr.plot(bvp_cont['pulse_rate_ts'].time / 60,
                   bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth"),  # noqa: E501
                   color=color_pr, linewidth=0.5, label='PR')
    ax_top_pr.set_ylabel('PR [1/min]')
    ax_top_pr.set_zorder(0)
    ax_top_pr.autoscale(enable=True, tight=True)

    ax_top.set_title("Blood volume pulse amplitude and pulse rate ("
                     +source+" | "+detector+")", fontweight='bold')
    lines_bvpa, labels_bvpa = ax_top.get_legend_handles_labels()
    lines_pr, labels_pr = ax_top_pr.get_legend_handles_labels()
    legend = ax_top.legend(lines_bvpa + lines_pr, labels_bvpa + labels_pr,
                       facecolor="white", framealpha=1,#
                       loc="upper right")
    legend.set_zorder(10)

    ax_top.xaxis.set_major_locator(MaxNLocator(nbins=14, prune=None))
    ax_top.patch.set_visible(False)

    # ----- Coherence-Scalogram
    ax_bottom_left = fig.add_subplot(gs[1, 0])

    # Color bar
    levels = np.linspace(0, 1, 50)
    cmap = cmap_parula()
    cf = ax_bottom_left.contourf(time, np.log2(freq), WCT, levels=levels,
                                 cmap=cmap, vmin=0, vmax=1)

    divider = make_axes_locatable(ax_bottom_left)
    cax = divider.append_axes("right", size="3%", pad=0.03)

    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_ticks(np.arange(0, 1.01, 0.2))
    # cbar.set_label('Coherence', rotation=90, labelpad=10)

    ax_bottom_left.set_title("Wavelet coherence (BVPA–PR coupling)", fontweight='bold')
    ax_bottom_left.set_xlabel("Time [min]")
    ax_bottom_left.xaxis.set_major_locator(MaxNLocator(nbins=14, prune=None))
    yticks = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax_bottom_left.set_yticks(np.log2(yticks))
    ax_bottom_left.set_yticklabels([f"{v:g}" for v in yticks])
    ax_bottom_left.set_ylim(np.log2(freq.min()), np.log2(2))
    ax_bottom_left.set_ylabel("Frequency [Hz]")

    # Cone of Influence (COI)
    coi = np.asarray(coi)
    coi_freq = 1.0 / coi
    ax_bottom_left.fill_between(
        time,
        np.log2(freq.min()) * np.ones_like(time),
        np.log2(coi_freq),
        facecolor="grey",
        alpha=0.6,
        edgecolor="k",
        linewidth=0.2,)

    # Significance
    if not significance == [0]:
        sig_arr = np.asarray(significance)

        if sig_arr.ndim == 1 and sig_arr.size == WCT.shape[0]:
            sig2d = sig_arr[:, None] * np.ones_like(WCT)
            ax_bottom_left.contour(time, np.log2(freq), WCT - sig2d,
                                   levels=[0], linewidths=1.0)

        elif sig_arr.ndim == 2 and sig_arr.shape == WCT.shape:
            ax_bottom_left.contour(time, np.log2(freq), WCT - sig_arr,
                                   levels=[0], linewidths=1.0)

    # Phase-Arrows
    TT, FF = np.meshgrid(time, freq)

    inside_coi = FF >= coi_freq[np.newaxis, :]
    strong_coh = WCT >= coherence_thresh
    mask = inside_coi & strong_coh

    ti = np.arange(0, n, arrow_step_time)
    si = np.arange(0, freq.size, arrow_step_period)
    T_sub = TT[np.ix_(si, ti)]
    P_sub = FF[np.ix_(si, ti)]
    phi_sub = aWCT[np.ix_(si, ti)]
    mask_sub = mask[np.ix_(si, ti)]

    U = np.cos(phi_sub)
    V = np.sin(phi_sub)

    ax_bottom_left.quiver(T_sub[mask_sub], np.log2(P_sub[mask_sub]),
                        U[mask_sub], V[mask_sub],
                        angles="uv", pivot="mid",
                        scale_units="width", scale=100,
                        width=0.001, headwidth=6, headlength=7)

    # ----- Means of coherence and phase over time
    S12_real = np.real(S12)
    S12_img = np.imag(S12)
    mean_help = np.sqrt((np.mean(S12_real, axis=1) **2)+(np.mean(S12_img, axis=1) **2))
    mean_coherence = np.abs(mean_help) ** 2 / np.mean((S1 * S2), axis=1)
    mean_phase = np.rad2deg(np.angle(np.mean(np.exp(1j * aWCT), axis=1)))

    freq_has_arrows = np.any(mask_sub, axis=1)
    freq_plot = np.log2(freq[si][freq_has_arrows])
    mean_phase_plot = mean_phase[si][freq_has_arrows]

    ax_bottom_right = fig.add_subplot(gs[1, 1])
    ax_bottom_right.plot(mean_coherence, np.log2(freq),
                         color=cmap(0.8), linewidth=2, label="Mean coherence")
    ax_bottom_right.set_xticks(np.arange(0, 1.0001, 0.2))
    ax_bottom_right.set_xlim(0, 1.001)
    ax_bottom_right.set_ylabel('')
    ax_bottom_right.set_yticks(np.log2(yticks))
    ax_bottom_right.set_yticklabels([f"{v:g}" for v in yticks])
    ax_bottom_right.yaxis.tick_right()
    ax_bottom_right.yaxis.set_label_position("right")
    ax_bottom_right.set_ylim(np.log2(freq.min()), np.log2(2))
    ax_bottom_right.set_zorder(1)

    ax_bottom_right_phase = ax_bottom_right.twiny()
    ax_bottom_right_phase.barh(freq_plot, mean_phase_plot,
                               color='k', linewidth=1, label="Mean phase [deg]",
                               height=(freq_plot[0]-freq_plot[1])*0.5)
    ax_bottom_right_phase.set_zorder(0)
    ax_bottom_right_phase.set_xlim(-181, 181)
    ax_bottom_right_phase.set_xticks(np.arange(-180, 181, 90))

    ax_bottom_right.patch.set_visible(False)

    lines_coh, labels_coh = ax_bottom_right.get_legend_handles_labels()
    lines_ph, labels_ph = ax_bottom_right_phase.get_legend_handles_labels()
    leg_coh = ax_bottom_right.legend(lines_coh, labels_coh,
                                    facecolor="white", framealpha=1,
                                    loc="lower right", fontsize=7)
    leg_coh.set_zorder(10)
    leg_ph = ax_bottom_right_phase.legend(lines_ph, labels_ph,
                                    facecolor="white", framealpha=1,
                                    loc="upper right", fontsize=7,
                                    handleheight=0.3)
    leg_ph.set_zorder(10)

    plt.show()
