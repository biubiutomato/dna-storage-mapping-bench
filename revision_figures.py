#!/usr/bin/env python3
"""Updated Figs. 1, 4 and 7 for the camera-ready version (adds CommaRotate, the capacity bound, and the strict comma decoder)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dna_storage_bench as B
import plots as P

plt.rcParams["font.family"] = "DejaVu Sans"  # same as the original figures

OUT = B.OUT
FIG = B.FIG_DIR
ORDER6 = P.ORDER + ["MarkerResync"]
CAP = 1.9826  # log2(#valid 150-mers)/150 for GC in [0.40, 0.60], HP <= 3 (capacity.py)


def fig1():
    cons = pd.read_csv(f"{OUT}/rev_constraint_summary.csv")
    comma = pd.read_csv(f"{OUT}/rev_comma_constraint.csv")
    df = pd.concat([cons, comma], ignore_index=True).set_index("codec").loc[ORDER6]
    x = np.arange(len(ORDER6))
    fig, (a, b) = plt.subplots(1, 2, figsize=(P.W, 1.85))
    for ax, col, lab, lim in ((a, "nominal_bits_per_nt", "Bits/nt", 2.25), (b, "valid_rate", "Valid rate", 1.12)):
        ax.bar(x, df[col], 0.66, color=[P.COLOR[c] for c in ORDER6], edgecolor="white", linewidth=1)
        for xi, v in zip(x, df[col]):
            ax.text(xi, v + lim * 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4)
        ax.set_xticks(x)
        ax.set_xticklabels([P.NAME[c] for c in ORDER6], rotation=40, ha="right")
        ax.set_ylabel(lab)
        ax.set_ylim(0, lim)
        ax.grid(axis="x", visible=False)
    a.axhline(CAP, color=P.INK, ls="--", lw=0.8)
    a.text(len(ORDER6) - 0.5, CAP + 0.05, "constrained\ncapacity 1.98", ha="right", va="bottom", fontsize=5.4)
    fig.tight_layout(w_pad=1.0)
    P._save(fig, f"{FIG}/fig1_density_valid.png")


def fig4():
    err = pd.concat([pd.read_csv(f"{OUT}/rev_error_summary.csv"), pd.read_csv(f"{OUT}/rev_comma_error.csv")], ignore_index=True)
    old = P.ORDER[:]
    P.ORDER[:] = ORDER6
    try:
        P.fig4(err, FIG)
    finally:
        P.ORDER[:] = old


def fig7():
    cm = pd.read_csv(f"{OUT}/rev_comma_control.csv")
    chans = [("substitution", "Substitution"), ("insertion", "Insertion"), ("deletion", "Deletion")]
    fig, axes = plt.subplots(1, 3, figsize=(P.W, 1.6), sharey=True)
    for ax, (ch, title) in zip(axes, chans):
        for n in ["Naive2Bit", "GoldmanRotate", "MarkerResync"]:
            d = cm[(cm.codec == n) & (cm.channel == ch)].sort_values("p")
            ax.plot(d["p"], d["block_recovery"], marker=P.MARK[n], ls=P.LS[n], color=P.COLOR[n], label=P.NAME[n])
        d = cm[(cm.codec == "MarkerResync") & (cm.channel == ch)].sort_values("p")
        ax.plot(d["p"], d["block_recovery_strict"], marker="X", ls="--", color="#5b2a86", label="CommaRotate-strict")
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xticks([0, 0.001, 0.01])
        ax.set_xticklabels(["0", ".001", ".01"])
        ax.set_xlabel("$p$")
        ax.set_ylim(0.2, 1.03)
    axes[0].set_ylabel("Block recovery")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=0.4)
    P._save(fig, f"{FIG}/fig7_marker.png")


if __name__ == "__main__":
    fig1(); fig4(); fig7()
    print("figures written to", FIG)
