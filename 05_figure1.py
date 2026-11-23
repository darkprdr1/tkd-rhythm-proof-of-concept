"""Figure 1: proof-of-concept match on an effective-time axis. Run after 02_exchanges.py.

A  Onset raster per round (rows = athletes) on effective time; grey bands mark
   removed stoppages (number = length, s); dark links join onsets of one exchange
   (successive onsets of either athlete <= theta apart, no stoppage between).
B  Each athlete's inter-action intervals per round (stoppage-spanning intervals
   excluded): median and interquartile range, CV and median above.
C  Intervals of the combined series (both athletes, simultaneous onsets merged) on
   a log axis in equal log-width bins, fitted two-component mixture density,
   provisional threshold theta and the range of its alternative estimates.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from common import FIG, OUT

ev = pd.read_csv(OUT / "events_effective.csv", dtype={"code": str})
iai = pd.read_csv(OUT / "iai_series.csv")
stops = pd.read_csv(OUT / "stoppages.csv")
qc = pd.read_csv(OUT / "round_qc.csv")
X = json.load(open(OUT / "exchanges.json"))
THETA = X["theta"]
TLO, THI = X["theta_range"]

COL = {"A": "#2a78d6", "B": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
STOPFILL, LINK = "#d9d8d3", "#3d3c39"

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(7.0, 6.9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45], width_ratios=[1.3, 1],
                      hspace=0.68, wspace=0.3, left=0.085, right=0.985, top=0.865, bottom=0.075)
axR = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
axB = fig.add_subplot(gs[2, 0])
axC = fig.add_subplot(gs[2, 1])
ROWY = {"A": 1.0, "B": 0.0}
XMAX = 135

# ---------------- Panel A ----------------
acts = ev[ev.event == "Action"]
for k, rnd in enumerate((1, 2)):
    ax = axR[k]
    g = ev[ev["round"] == rnd]
    rend = qc.loc[qc["round"] == rnd, "effective_s"].iloc[0]
    st = stops[stops["round"] == rnd].sort_values("eff_at").reset_index(drop=True)
    last_x, level = -99, 0
    for _, s in st.iterrows():
        ax.add_patch(plt.Rectangle((s.eff_at - 0.45, -0.5), 0.9, 2.05, facecolor=STOPFILL, lw=0, zorder=0))
        level = 1 - level if s.eff_at - last_x < 4.5 else 0
        ax.text(s.eff_at, 1.60 + 0.30 * level, f"{s.dur:.1f}", ha="center", va="bottom",
                fontsize=6.3, color=MUTED, zorder=3)
        last_x = s.eff_at
    ax.plot([rend, rend], [-0.45, 1.45], color=INK2, lw=0.8, zorder=1)
    ax.text(rend + 0.8, -0.45, "end", fontsize=6.3, color=INK2, va="bottom", ha="left")

    a = acts[acts["round"] == rnd].sort_values(["eff", "athlete"]).reset_index(drop=True)
    merged = a.groupby("eff", as_index=False).agg(seg=("seg", "first"), who=("athlete", lambda s: "".join(sorted(s))))
    for i in range(1, len(merged)):
        t0, t1 = merged.eff[i - 1], merged.eff[i]
        if t1 - t0 <= THETA and merged.seg[i] == merged.seg[i - 1]:
            w0, w1 = merged.who[i - 1], merged.who[i]
            if w0 == w1 and len(w0) == 1:
                yy = ROWY[w0] + (0.17 if w0 == "A" else -0.17)
                ax.plot([t0 - 0.6, t1 + 0.6], [yy, yy], color=LINK, lw=1.1, solid_capstyle="butt", zorder=2)
            else:
                ax.plot([t0, t1], [ROWY[w0] if len(w0) == 1 else 1.0, ROWY[w1] if len(w1) == 1 else 0.0],
                        color=LINK, lw=0.9, zorder=2)
    for _, r in merged[merged.who == "AB"].iterrows():
        ax.plot([r.eff, r.eff], [0, 1], color=LINK, lw=0.9, zorder=2)

    for ath in ("A", "B"):
        aa = g[(g.event == "Action") & (g.athlete == ath)].sort_values("eff")
        y = ROWY[ath]
        ax.scatter(aa.eff, np.full(len(aa), y), s=17, color=COL[ath], edgecolor="white", linewidth=0.6, zorder=3)
        pts_col = "pts_A" if ath == "A" else "pts_B"
        sc = aa[aa[pts_col].notna()]
        ax.scatter(sc.eff, np.full(len(sc), y + 0.36), marker="*", s=70, color=COL[ath],
                   edgecolor=INK, linewidth=0.5, zorder=4)
        for _, r in sc.iterrows():
            ax.text(r.eff + 1.1, y + 0.36, f"{int(r[pts_col])}", fontsize=6.3, color=INK, va="center", ha="left", zorder=4)
        pen = g[(g.event == "Marker") & (g.athlete == ath)]
        ax.scatter(pen.eff, np.full(len(pen), y - 0.36), marker="x", s=20, color=INK2, linewidth=0.9, zorder=4)
    ax.set_xlim(-1.5, XMAX)
    ax.set_ylim(-0.6, 2.05)
    ax.set_yticks([1, 0], ["Athlete A", "Athlete B"])
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(range(0, 121, 20))
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=-1)
    ax.set_axisbelow(True)
    exr = [e for e in X["exchanges"] if e["round"] == rnd and abs(e["theta"] - THETA) < 1e-6][0]
    nA, nB = int((a.athlete == "A").sum()), int((a.athlete == "B").sum())
    ax.text(0.0, 1.10, f"Round {rnd}", transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=INK, ha="left", va="bottom")
    ax.text(1.0, 1.10, f"actions A {nA}, B {nB}   ·   exchanges {exr['exchanges']} "
            f"({exr['in_exch']*100:.0f}% of onsets)   ·   stoppages {len(st)} ({st.dur.sum():.1f} s removed)"
            f"   ·   effective time {rend:.1f} s",
            transform=ax.transAxes, fontsize=6.6, color=INK2, ha="right", va="bottom")
axR[1].set_xlabel("Effective time within round (s; stoppages removed)")
axR[0].set_title("A   Action onsets and exchanges on the effective-time axis", pad=24)

handles = [
    Line2D([], [], marker="o", ls="", ms=4.5, color=INK2, mec="white", mew=0.5, label="action onset"),
    Line2D([], [], color=LINK, lw=0.9, label=f"exchange link (onsets ≤ {THETA:.2f} s apart)"),
    Line2D([], [], marker="*", ls="", ms=8.5, color="#bdbcb6", mec=INK, mew=0.5, label="scoring action (points)"),
    Line2D([], [], marker="x", ls="", ms=4.5, color=INK2, mew=0.9, label="gam-jeom (penalised athlete)"),
    Patch(facecolor=STOPFILL, edgecolor="none", label="stoppage removed (length, s)"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.998), ncol=3,
           columnspacing=1.1, handletextpad=0.4, handlelength=1.6)

# ---------------- Panel B ----------------
rng = np.random.default_rng(7)
for (ath, rnd), x in zip([("A", 1), ("A", 2), ("B", 1), ("B", 2)], [0, 1, 2.35, 3.35]):
    v = iai[(iai.series == ath) & (iai["round"] == rnd) & (~iai.spans_stop)].iai.values
    axB.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=12, color=COL[ath], edgecolor="white", linewidth=0.5, zorder=3)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    axB.add_patch(plt.Rectangle((x - 0.3, q1), 0.6, q3 - q1, fill=False, edgecolor=INK, lw=0.8, zorder=4))
    axB.plot([x - 0.3, x + 0.3], [med, med], color=INK, lw=1.6, zorder=5)
    axB.text(x, 13.6, f"CV {v.std(ddof=1) / v.mean():.2f}", ha="center", fontsize=6.8, color=INK)
    axB.text(x, 12.85, f"Mdn {med:.2f}", ha="center", fontsize=6.3, color=INK2)
axB.set_xticks([0, 1, 2.35, 3.35], ["A · R1", "A · R2", "B · R1", "B · R2"])
axB.set_xlim(-0.55, 3.9)
axB.set_ylim(0, 14.4)
axB.set_yticks(range(0, 13, 2))
axB.set_ylabel("Inter-action interval (s)")
axB.tick_params(axis="x", length=0)
axB.yaxis.grid(True, color=GRID, lw=0.5)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.set_title("B   Each athlete's intervals (conditional on the dyad)", pad=8)

# ---------------- Panel C ----------------
bv = iai[(iai.series == "AB") & (~iai.spans_stop)].iai.values
N_BINS = 21
edges = np.logspace(np.log10(0.05), np.log10(8), N_BINS + 1)
binw = np.log10(edges[1]) - np.log10(edges[0])
axC.axvspan(TLO, THI, color="#efeee9", lw=0, zorder=0)
axC.hist(np.clip(bv, 0.05, None), bins=edges, color="#8f8e88", edgecolor="white", linewidth=0.6, zorder=2)
axC.plot(np.array(X["mix_grid"]), np.array(X["mix_dens"]) * len(bv) * binw, color=INK, lw=1.1, zorder=3)
axC.axvline(THETA, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=4)
axC.set_xscale("log")
axC.set_xlim(0.05, 8)
axC.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
axC.minorticks_off()
axC.set_xlabel(f"Interval between successive onsets (s, log scale;\nbin width {binw:.3f} log10 units)")
axC.set_ylabel("Intervals (n)")
axC.yaxis.grid(True, color=GRID, lw=0.5)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
YM = 23
axC.set_ylim(0, YM)
axC.text(THETA * 0.9, YM * 0.97, f"provisional θ = {THETA:.2f} s\nrange {TLO:.2f}–{THI:.2f}", fontsize=6.5,
         color=INK, va="top", ha="right")
axC.text(0.115, YM * 0.40, "shorter\nmode", fontsize=6.5, color=INK2, ha="center")
axC.text(5.0, YM * 0.82, "longer\nmode", fontsize=6.5, color=INK2, ha="center")
axC.set_title("C   Combined (dyadic) series", pad=8)

for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(FIG / f"Figure1_R2.{ext}", **kw)
fig.savefig(FIG / "Figure1_R2_300dpi.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
im = Image.open(FIG / "Figure1_R2_300dpi.tiff").convert("RGB")
im.save(FIG / "Figure1_R2_300dpi.tiff", compression="tiff_lzw", dpi=(300, 300))
print("figure", im.size, "bin width (log10 units)", round(binw, 4), "bins", N_BINS)
