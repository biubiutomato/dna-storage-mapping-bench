"""Paper figures. Internal codec names are kept in the CSVs; figures use the
display names used in the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = {
    "Naive2Bit": "Naive2Bit",
    "Screened2Bit": "Screened2Bit",
    "Block6to4": "Block6to4",
    "GoldmanRotate": "TernaryRotate",
    "Church1Bit": "Pair1Bit",
    "MarkerResync": "CommaRotate",
}
COLOR = {
    "Naive2Bit": "#2a78d6",
    "Screened2Bit": "#eb6834",
    "Block6to4": "#1baf7a",
    "GoldmanRotate": "#eda100",
    "Church1Bit": "#e87ba4",
    "MarkerResync": "#008300",
}
MARK = {"Naive2Bit": "o", "Screened2Bit": "s", "Block6to4": "^",
        "GoldmanRotate": "D", "Church1Bit": "v", "MarkerResync": "P"}
LS = {"Naive2Bit": "-", "Screened2Bit": "--", "Block6to4": "-.",
      "GoldmanRotate": "-", "Church1Bit": ":", "MarkerResync": "-"}
ORDER = ["Naive2Bit", "Screened2Bit", "Block6to4", "GoldmanRotate", "Church1Bit"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300,
    "lines.linewidth": 1.4, "lines.markersize": 3.8,
})
W = 3.45  # IEEE single-column width in inches


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1(cons_df, out):
    sub = cons_df.set_index("codec").loc[ORDER]
    x = np.arange(len(ORDER))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.75))
    for ax, col, lab, lim in ((a, "nominal_bits_per_nt", "Bits/nt", 2.2),
                              (b, "valid_rate", "Valid rate", 1.1)):
        ax.bar(x, sub[col], 0.66, color=[COLOR[c] for c in ORDER], edgecolor="white", linewidth=1)
        for xi, v in zip(x, sub[col]):
            ax.text(xi, v + lim * 0.015, f"{v:.2f}" if col.startswith("nom") else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.6, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[c] for c in ORDER], rotation=40, ha="right")
        ax.set_ylabel(lab)
        ax.set_ylim(0, lim)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    _save(fig, f"{out}/fig1_density_valid.png")


def _hist(samples, bins, out, xlabel, fname, band=None, vline=None, ymax=None, skip=()):
    fig, ax = plt.subplots(figsize=(W, 1.9))
    if band:
        ax.axvspan(*band, color="#ecebe4", zorder=0, label="Allowed window")
    for n in ORDER:
        if n in skip:
            continue
        ax.hist(samples[n], bins=bins, histtype="step", density=True, lw=1.3,
                ls=LS[n], color=COLOR[n], label=NAME[n])
    if vline is not None:
        ax.axvline(vline, color=INK, ls="--", lw=0.8, label="Limit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if ymax:
        ax.set_ylim(0, ymax)
    if skip:
        ax.annotate(" and ".join(NAME[s] for s in skip) + ":\nevery oligo at GC = 0.5",
                    xy=(0.5, ymax * 0.97), xytext=(0.62, ymax * 0.55), fontsize=6, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
        ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (1, 1)))
    ax.legend(ncol=1 if skip else 2, loc="upper left" if skip else "upper right")
    fig.tight_layout()
    _save(fig, f"{out}/{fname}")


def fig2(gc_samples, out, lo=0.40, hi=0.60):
    step = 1 / 75  # one base at L = 150; bins centred on 0.5
    bins = 0.5 + step * (np.arange(-20, 21) - 0.5)
    _hist(gc_samples, bins, out, "GC fraction", "fig2_gc_hist.png", band=(lo, hi),
          ymax=24, skip=("Block6to4", "Church1Bit"))


def fig3(hp_samples, out, hp_max=3):
    m = max(max(v) for v in hp_samples.values())
    _hist(hp_samples, np.arange(0.5, m + 1.5, 1), out,
          "Longest homopolymer per oligo (nt)", "fig3_hp_hist.png", vline=hp_max + 0.5)


def _grid(err_df, col, ylabel, ylim, out, fname, legend_loc):
    chans = [("substitution", "Substitution"), ("insertion", "Insertion"),
             ("deletion", "Deletion"), ("mixed_sid", "Mixed SID")]
    fig, axes = plt.subplots(2, 2, figsize=(W, 2.9), sharex=True, sharey=True)
    for ax, (ch, title) in zip(axes.ravel(), chans):
        for n in ORDER:
            d = err_df[(err_df.codec == n) & (err_df.channel == ch)]
            ax.plot(d["p"], d[col], marker=MARK[n], ls=LS[n], color=COLOR[n], label=NAME[n])
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xticks([0, 0.001, 0.01, 0.05])
        ax.set_xticklabels(["0", "0.001", "0.01", "0.05"])
        ax.set_ylim(*ylim)
    for ax in axes[1]:
        ax.set_xlabel("Per-base error rate $p$")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    _save(fig, f"{out}/{fname}")


def fig4(err_df, out):
    _grid(err_df, "exact_recovery", "Exact recovery", (-0.03, 1.04), out,
          "fig4_recovery.png", "upper right")


def fig5(err_df, out):
    _grid(err_df, "bit_similarity", "Bit similarity", (0.45, 1.02), out,
          "fig5_bit_sim.png", "lower left")


def fig6(sens_df, out):
    d = sens_df[sens_df.window == "GC40-60_HP3"]
    fig, ax = plt.subplots(figsize=(W, 1.8))
    for n in ["Naive2Bit", "Block6to4", "GoldmanRotate", "Church1Bit"]:
        s = d[d.codec == n].sort_values("L")
        ax.plot(s["L"], s["valid_rate"], marker=MARK[n], ls=LS[n], color=COLOR[n], label=NAME[n])
    ax.set_xticks([100, 150, 200])
    ax.set_xlabel("Oligo length $L$ (nt)")
    ax.set_ylabel("Valid rate")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, loc="center right")
    fig.tight_layout()
    _save(fig, f"{out}/fig6_length.png")


def fig7(ctrl_df, out):
    codecs = ["Naive2Bit", "GoldmanRotate", "MarkerResync"]
    chans = [("substitution", "Substitution"), ("insertion", "Insertion"), ("deletion", "Deletion")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.75), sharey=True)
    for ax, (ch, title) in zip(axes, chans):
        for n in codecs:
            d = ctrl_df[(ctrl_df.codec == n) & (ctrl_df.channel == ch)]
            ax.plot(d["p"], d["block_recovery"], marker=MARK[n], ls=LS[n], color=COLOR[n], label=NAME[n])
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xticks([0, 0.001, 0.01])
        ax.set_xticklabels(["0", "0.001", "0.01"])
        ax.set_ylim(0.2, 1.03)
        ax.set_xlabel("$p$")
    axes[0].set_ylabel("Block recovery")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.4)
    _save(fig, f"{out}/fig7_marker.png")
