
# Lager für den Summenplot, damit ich ihn aus der Datei "bvp_wav_ana_v12"
# heruasnehemn kann.


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

from cedalion.sigproc.frequency import sampling_rate
from cedalion.dataclasses.bvp_container import BVP_Container

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
    cbar.set_label('Coherence', rotation=90, labelpad=10)

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
