"""Shared loading and helper functions for the proof-of-concept analysis.

Event log columns (data/event_log_coder1.csv):
  round, round_time_s (time since the round's Shi-jak, stoppages included),
  event (Round start / Round end / Stop start / Stop end / Action /
  Continuation / Marker), athlete, code, combo, pts_A, pts_B, reason, notes

Definitions (Supplementary Section S2):
  effective time  = round time minus all completed stoppages before the event
  segment         = stretch of uninterrupted play (increments at each Stop end)
  interval        = difference in effective time between successive onsets of a
                    series within the same segment; an interval whose two onsets
                    lie in different segments spans a stoppage
  combined series = both athletes' onsets in effective-time order, onsets with
                    identical timestamps merged into one
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
FIG = HERE / "fig"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

FPS = 30.0                 # nominal frame rate of the broadcast footage
FRAME = 1.0 / FPS          # one frame, s
THETA_FALLBACK = 0.87      # provisional exchange threshold (s), derived in 02_exchanges.py


def load_events(path=DATA / "event_log_coder1.csv"):
    """Return the event log with effective time (eff) and segment (seg) recomputed."""
    df = pd.read_csv(path, dtype={"code": str, "athlete": str, "combo": str, "reason": str})
    rows = []
    for rnd, g in df.groupby("round", sort=True):
        g = g.sort_values("round_time_s", kind="stable")
        paused, seg, stop_open = 0.0, 0, None
        for _, r in g.iterrows():
            d = r.to_dict()
            if r.event == "Stop start":
                stop_open = r.round_time_s
            elif r.event == "Stop end":
                d["stop_dur"] = round(r.round_time_s - stop_open, 2)
                paused += r.round_time_s - stop_open
                seg += 1
                stop_open = None
            d["eff"] = round(r.round_time_s - paused, 2)
            d["seg"] = seg
            rows.append(d)
    return pd.DataFrame(rows)


def actions(ev):
    return ev[ev.event == "Action"].copy()


def round_length(ev, rnd):
    return float(ev[(ev["round"] == rnd) & (ev.event == "Round end")].eff.iloc[0])


def stoppages(ev):
    out = []
    for rnd, g in ev.groupby("round"):
        s0 = g[g.event == "Stop start"].reset_index(drop=True)
        s1 = g[g.event == "Stop end"].reset_index(drop=True)
        for i in range(len(s0)):
            out.append(dict(round=rnd, no=i + 1, eff_at=float(s0.eff[i]),
                            dur=round(float(s1.round_time_s[i] - s0.round_time_s[i]), 2),
                            reason=s0.reason[i], seg_before=int(s0.seg[i])))
    return pd.DataFrame(out)


def merged_onsets(acts_round):
    """Combined series of one round: identical timestamps merged into one onset."""
    m = (acts_round.groupby("eff", as_index=False)
         .agg(seg=("seg", "first"), who=("athlete", lambda s: "".join(sorted(s))))
         .sort_values("eff").reset_index(drop=True))
    return m


def interval_table(sub):
    """Intervals between successive rows of a series (sorted by effective time)."""
    sub = sub.sort_values("eff", kind="stable").reset_index(drop=True)
    out = []
    for i in range(1, len(sub)):
        a, b = sub.loc[i - 1], sub.loc[i]
        out.append(dict(t_from=a.eff, t_to=b.eff, iai=round(b.eff - a.eff, 2),
                        spans_stop=bool(a.seg != b.seg), seg_from=int(a.seg)))
    return pd.DataFrame(out)


def describe(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return dict(n=n)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=n, median=float(med), q1=float(q1), q3=float(q3), min=float(x.min()),
                max=float(x.max()), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                cv=float(x.std(ddof=1) / x.mean()), tempo=float(60 * n / x.sum()))


def boot_ci(x, stat, n_boot=10000, seed=1):
    """Percentile bootstrap 95% CI of a statistic of the intervals (resampled i.i.d.)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    vals = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cv(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / x.mean()


def segment_exchanges(m, theta):
    """Split a combined series into units; exchanges are runs of >= 2 onsets whose
    successive intervals are all <= theta within one segment."""
    groups, cur = [], [0]
    for i in range(1, len(m)):
        if (m.eff[i] - m.eff[i - 1] <= theta) and (m.seg[i] == m.seg[i - 1]):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups, [g for g in groups if len(g) >= 2]


def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measure."""
    d = np.vstack([x, y]).T.astype(float)
    n, k = d.shape
    gm = d.mean()
    ms_r = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))
