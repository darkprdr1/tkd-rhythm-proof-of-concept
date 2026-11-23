"""Reanalyses requested in the second review round. Run after 02_exchanges.py.

1. Dyadic dependence of the two athletes' interval series, and the individual
   components derived conditionally on the dyad's exchange structure.
2. Within-round drift D: onset counts per third of effective time with exact
   confidence intervals, and a log-linear Poisson rate trend fitted to the whole round.
3. Censoring by stoppages: Kaplan-Meier estimates that treat an interval interrupted
   by a stoppage (or by the end of the round) as right-censored instead of excluding it.
"""
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

from common import (OUT, actions, describe, load_events, merged_onsets, round_length,
                    segment_exchanges, stoppages)

ev = load_events()
acts = actions(ev)
stops = stoppages(ev)
X = json.load(open(OUT / "exchanges.json"))
THETA = X["theta"]
res = {}

# ------------------------------------------------------------------ 1. dyad
dy = {}
short_strict = short_all = 0
pair_A, pair_B = [], []          # intervals between successive two-athlete exchanges, per athlete
for rnd, sub in acts.groupby("round"):
    m = merged_onsets(sub)
    groups, ex = segment_exchanges(m, THETA)
    for i in range(1, len(m)):
        if m.seg[i] == m.seg[i - 1] and (m.eff[i] - m.eff[i - 1]) < 0.75:
            short_all += 1
            short_strict += int(len(m.who[i]) == 1 and len(m.who[i - 1]) == 1 and m.who[i] != m.who[i - 1])

    # each athlete's onsets: inside a two-athlete exchange, or not
    in_joint = {}
    joint = []
    for g in ex:
        who = "".join(m.who[i] for i in g)
        if "A" in who and "B" in who:
            joint.append(g)
            for i in g:
                in_joint[i] = True
    r = dict(n_exchanges=len(ex), n_joint=len(joint))
    for ath in "AB":
        idx = [i for i in range(len(m)) if ath in m.who[i]]
        answered = sum(1 for i in idx if in_joint.get(i))
        r[f"onsets_{ath}"] = len(idx)
        r[f"in_joint_exchange_{ath}"] = answered / len(idx)
        r[f"unanswered_{ath}"] = len(idx) - answered
        # longest run of this athlete's consecutive onsets not answered within theta
        run = best = 0
        for i in range(len(m)):
            if m.who[i] == ath and not in_joint.get(i):
                run += 1
                best = max(best, run)
            elif ath not in m.who[i] or in_joint.get(i):
                run = 0
        r[f"longest_unanswered_run_{ath}"] = best
        # this athlete's own intervals: both ends inside a two-athlete exchange?
        own = [i for i in idx]
        shared = total = 0
        for a, b in zip(own[:-1], own[1:]):
            if m.seg[a] == m.seg[b]:
                total += 1
                shared += int(bool(in_joint.get(a)) and bool(in_joint.get(b)))
        r[f"intervals_{ath}"] = total
        r[f"intervals_shared_{ath}"] = shared / total

    # opening and response lag within two-athlete exchanges
    opened = {"A": 0, "B": 0, "AB": 0}
    lag = {"A": [], "B": []}       # lag[x] = lags with which x answered the opponent's opening onset
    for g in joint:
        w0 = m.who[g[0]]
        opened[w0] += 1
        if len(w0) == 1:
            other = "B" if w0 == "A" else "A"
            first_other = next(i for i in g if other in m.who[i])
            lag[other].append(float(m.eff[first_other] - m.eff[g[0]]))
    r["opened"] = {k: v / len(joint) for k, v in opened.items()}
    r["response_lag_median"] = {k: (float(np.median(v)) if v else None) for k, v in lag.items()}
    r["response_lag_n"] = {k: len(v) for k, v in lag.items()}

    # the same between-exchange gap seen from each athlete: first own onset in successive joint exchanges
    for g0, g1 in zip(joint[:-1], joint[1:]):
        if m.seg[g0[0]] != m.seg[g1[0]]:
            continue
        a0 = next(m.eff[i] for i in g0 if "A" in m.who[i]); a1 = next(m.eff[i] for i in g1 if "A" in m.who[i])
        b0 = next(m.eff[i] for i in g0 if "B" in m.who[i]); b1 = next(m.eff[i] for i in g1 if "B" in m.who[i])
        pair_A.append(a1 - a0)
        pair_B.append(b1 - b0)
    dy[f"R{rnd}"] = r

pair_A, pair_B = np.array(pair_A), np.array(pair_B)
dy["paired_exchange_intervals"] = dict(
    n=len(pair_A), pearson_r=float(np.corrcoef(pair_A, pair_B)[0, 1]),
    spearman_rho=float(stats.spearmanr(pair_A, pair_B)[0]),
    median_abs_diff=float(np.median(np.abs(pair_A - pair_B))),
    max_abs_diff=float(np.abs(pair_A - pair_B).max()))
dy["short_lt_075"] = dict(n=short_all, cross_athlete_strict=short_strict)
res["dyad"] = dy

# ------------------------------------------------------------------ 2. drift


def exact_ratio_ci(n1, n3, e1=1.0, e3=1.0, level=0.95):
    """Exact (conditional binomial, Clopper-Pearson) CI for the rate ratio (n3/e3)/(n1/e1)."""
    n = n1 + n3
    a = 1 - level
    lo_p = stats.beta.ppf(a / 2, n3, n1 + 1) if n3 > 0 else 0.0
    hi_p = stats.beta.ppf(1 - a / 2, n3 + 1, n1) if n1 > 0 else 1.0
    f = e1 / e3
    return (float(lo_p / (1 - lo_p) * f), float(hi_p / (1 - hi_p) * f) if hi_p < 1 else float("inf"),
            float(stats.binomtest(n3, n, e3 / (e1 + e3)).pvalue))


def loglinear_trend(t, L):
    """Inhomogeneous Poisson process with rate exp(a + b t) on [0, L].
    Conditional on n, onset times have density b exp(b t) / (exp(b L) - 1).
    Returns the end-to-start rate ratio exp(b L) with a likelihood-ratio 95% CI and p value."""
    t = np.asarray(t, float)
    n = len(t)

    def ll(b):
        if abs(b) < 1e-9:
            return -n * np.log(L)
        return n * np.log(abs(b)) + b * t.sum() - n * np.log(abs(np.expm1(b * L)))

    opt = optimize.minimize_scalar(lambda b: -ll(b), bounds=(-0.2, 0.2), method="bounded")
    b = opt.x
    lr = 2 * (ll(b) - ll(0.0))
    crit = stats.chi2.ppf(0.95, 1)
    f = lambda x: 2 * (ll(b) - ll(x)) - crit
    lo = optimize.brentq(f, -0.2, b)
    hi = optimize.brentq(f, b, 0.2)
    return dict(slope_per_s=float(b), ratio_end_start=float(np.exp(b * L)),
                ci=[float(np.exp(lo * L)), float(np.exp(hi * L))], p=float(stats.chi2.sf(lr, 1)))


drift = {}
for rnd in (1, 2):
    L = round_length(ev, rnd)
    series = {a: acts[(acts["round"] == rnd) & (acts.athlete == a)].eff.values for a in "AB"}
    series["AB"] = merged_onsets(acts[acts["round"] == rnd]).eff.values
    for name, t in series.items():
        c = [int(((t >= k * L / 3) & (t < (k + 1) * L / 3)).sum()) for k in range(3)]
        c[2] += int((t >= L).sum())
        lo, hi, p = exact_ratio_ci(c[0], c[2])
        # thirds of the nominal 120 s official clock (boundaries at 40 and 80 s of effective time)
        cn = [int((t < 40).sum()), int(((t >= 40) & (t < 80)).sum()), int((t >= 80).sum())]
        lon, hin, pn = exact_ratio_ci(cn[0], cn[2], 40.0, L - 80.0)
        drift[f"{name}R{rnd}"] = dict(
            thirds=c, rates_per_min=[60 * v / (L / 3) for v in c], D=c[2] / c[0], D_ci=[lo, hi], D_p=p,
            thirds_nominal=cn, D_nominal=(cn[2] / (L - 80.0)) / (cn[0] / 40.0), D_nominal_ci=[lon, hin],
            trend=loglinear_trend(t, L))
# do the two athletes' Round 2 profiles differ? first vs last third, Fisher exact test
a, b = drift["AR2"]["thirds"], drift["BR2"]["thirds"]
drift["R2_A_vs_B_fisher_p"] = float(stats.fisher_exact([[a[0], a[2]], [b[0], b[2]]])[1])
res["drift"] = drift

# ------------------------------------------------------------------ 3. censoring


def km(times, observed):
    """Kaplan-Meier survival function; returns event times, S(t), and quantile/mean helpers."""
    times, observed = np.asarray(times, float), np.asarray(observed, bool)
    ut = np.unique(times[observed])
    s, S = 1.0, []
    for u in ut:
        at_risk = (times >= u).sum()
        d = ((times == u) & observed).sum()
        s *= 1 - d / at_risk
        S.append(s)
    return ut, np.array(S)


def km_quantile(ut, S, q):
    idx = np.where(S <= 1 - q)[0]
    return float(ut[idx[0]]) if len(idx) else float("nan")


def km_rmean(ut, S, tau):
    """Restricted mean up to tau (area under the survival curve)."""
    grid = np.concatenate([[0.0], ut[ut < tau], [tau]])
    surv = np.concatenate([[1.0], S[ut < tau]])
    return float(np.sum(np.diff(grid) * surv))


cens = {}
for rnd in (1, 2):
    L = round_length(ev, rnd)
    st = stops[stops["round"] == rnd]
    stop_at = {int(r.seg_before): float(r.eff_at) for r in st.itertuples()}   # segment -> effective time of the stop
    last_seg = int(ev[ev["round"] == rnd].seg.max())
    sub_all = acts[acts["round"] == rnd]
    series = {a: sub_all[sub_all.athlete == a].sort_values("eff", kind="stable").reset_index(drop=True) for a in "AB"}
    series["AB"] = merged_onsets(sub_all)
    for name, s in series.items():
        t, obs = [], []
        for i in range(len(s)):
            seg = int(s.seg[i])
            end = stop_at.get(seg, L)          # this stretch of play ends at a stoppage or at the round end
            if i + 1 < len(s) and int(s.seg[i + 1]) == seg:
                t.append(round(float(s.eff[i + 1] - s.eff[i]), 2)); obs.append(True)
            else:
                c = round(float(end - s.eff[i]), 2)
                if c > 0:
                    t.append(c); obs.append(False)
        t, obs = np.array(t), np.array(obs)
        ut, S = km(t, obs)
        comp = t[obs]
        tau = float(t.max())
        naive = describe(comp)
        cens[f"{name}R{rnd}"] = dict(
            n_complete=int(obs.sum()), n_censored=int((~obs).sum()),
            censored_times=[round(float(v), 2) for v in t[~obs]],
            naive_median=naive["median"], naive_q1=naive["q1"], naive_q3=naive["q3"],
            naive_mean=naive["mean"], naive_tempo=naive["tempo"],
            km_median=km_quantile(ut, S, 0.5), km_q1=km_quantile(ut, S, 0.25), km_q3=km_quantile(ut, S, 0.75),
            km_rmean=km_rmean(ut, S, tau), naive_rmean=float(np.minimum(comp, tau).mean()),
            exposure_tempo=float(60 * obs.sum() / t.sum()),
            share_long_censored=float((t[~obs] > naive["median"]).mean()) if (~obs).any() else None)
res["censoring"] = cens

# are the long intervals the ones that get interrupted? pooled individual series
allc, alln = [], []
for k, v in cens.items():
    if not k.startswith("AB"):
        allc += v["censored_times"]
res["censoring_summary"] = dict(n_censored_individual=len(allc), median_censored_time=float(np.median(allc)),
                                max_censored_time=float(np.max(allc)))

json.dump(res, open(OUT / "reanalysis.json", "w"), indent=1, default=float)

print("== dyad")
for k in ("R1", "R2"):
    print(k, json.dumps({a: (round(b, 2) if isinstance(b, float) else b) for a, b in dy[k].items()}, default=float))
print(dy["paired_exchange_intervals"], dy["short_lt_075"])
print("== drift")
for k, v in drift.items():
    if not isinstance(v, dict):
        print(k, round(v, 3)); continue
    print(f"{k:5s} thirds {v['thirds']} rates {[round(x,1) for x in v['rates_per_min']]} D {v['D']:.2f} CI {v['D_ci'][0]:.2f}-{v['D_ci'][1]:.2f} p {v['D_p']:.2f}"
          f" | nominal {v['thirds_nominal']} D {v['D_nominal']:.2f} CI {v['D_nominal_ci'][0]:.2f}-{v['D_nominal_ci'][1]:.2f}"
          f" | trend ratio {v['trend']['ratio_end_start']:.2f} CI {v['trend']['ci'][0]:.2f}-{v['trend']['ci'][1]:.2f} p {v['trend']['p']:.2f}")
print("== censoring")
for k, v in cens.items():
    print(f"{k:5s} complete {v['n_complete']} cens {v['n_censored']} naive median {v['naive_median']:.2f} KM median {v['km_median']:.2f} "
          f"| naive IQR {v['naive_q1']:.2f}-{v['naive_q3']:.2f} KM IQR {v['km_q1']:.2f}-{v['km_q3']:.2f} "
          f"| rmean naive {v['naive_rmean']:.2f} KM {v['km_rmean']:.2f} | tempo naive {v['naive_tempo']:.1f} exposure {v['exposure_tempo']:.1f} | cens {v['censored_times']}")
print(res["censoring_summary"])
