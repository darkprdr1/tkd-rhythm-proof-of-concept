"""Provisional exchange threshold, exchange segmentation, own-burst instability,
scoring-capable subseries. Run after 01_metrics.py.

The threshold theta is the trough of a two-component Gaussian mixture fitted to the
log10 intervals of the combined series (both rounds pooled). It is derived from the
same data it is then used to describe, and is therefore provisional; alternative
estimates are reported as a sensitivity range.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture

from common import (FRAME, OUT, actions, describe, interval_table, load_events, merged_onsets,
                    segment_exchanges)

ev = load_events()
acts = actions(ev)
FLOOR = 0.05   # intervals below 0.05 s are treated as effectively simultaneous on the log scale
SCORING_LETTERS = re.compile(r"[CcfFSsOTHRNX]")


def scoring_capable(code):
    c = (code or "").replace("pH", "H")
    c = re.sub(r"[bhpu]$", "", c)
    return bool(SCORING_LETTERS.search(c))


merged, per_round_iv, pool = {}, {}, []
for rnd, sub in acts.groupby("round"):
    m = merged_onsets(sub)
    merged[rnd] = m
    iv = interval_table(m)
    per_round_iv[rnd] = iv
    pool.append(iv[~iv.spans_stop].iai.values)
pool = np.concatenate(pool)


def gmm_fit(x):
    lx = np.log10(np.clip(x, FLOOR, None)).reshape(-1, 1)
    return GaussianMixture(2, n_init=20, random_state=1).fit(lx)


def gmm_trough(x):
    g = gmm_fit(x)
    mu = np.sort(g.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 4000).reshape(-1, 1)
    dens = np.exp(g.score_samples(grid))
    return float(10 ** grid[int(np.argmin(dens)), 0]), g


def gmm_equal_posterior(x):
    g = gmm_fit(x)
    mu = np.sort(g.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 4000).reshape(-1, 1)
    p = g.predict_proba(grid)
    lo = int(np.argmin(g.means_.ravel()))
    return float(10 ** grid[int(np.argmin(np.abs(p[:, lo] - 0.5))), 0])


def antimode_log(x, bw):
    lx = np.log10(np.clip(x, FLOOR, None))
    grid = np.linspace(lx.min() - 0.3, lx.max() + 0.3, 2000)
    kde = stats.gaussian_kde(lx, bw_method="silverman")
    kde.set_bandwidth(kde.factor * bw)
    d = kde(grid)
    mx = argrelextrema(d, np.greater)[0]
    if len(mx) < 2:
        return float("nan")
    top2 = sorted(mx[np.argsort(d[mx])[-2:]])
    return float(10 ** grid[top2[0] + int(np.argmin(d[top2[0]:top2[1] + 1]))])


def antimode_lin(x, bw, grid=np.linspace(0.05, 6, 2000)):
    kde = stats.gaussian_kde(x, bw_method="silverman")
    kde.set_bandwidth(kde.factor * bw)
    d = kde(grid)
    mx = argrelextrema(d, np.greater)[0]
    if len(mx) < 2:
        return float("nan")
    top2 = sorted(mx[np.argsort(d[mx])[-2:]])
    return float(grid[top2[0] + int(np.argmin(d[top2[0]:top2[1] + 1]))])


theta_raw, GMM = gmm_trough(pool)
theta = round(theta_raw, 2)
checks = {
    "gmm_trough_pooled": theta_raw,
    "gmm_equal_posterior_pooled": gmm_equal_posterior(pool),
    "kde_log_bw0.75": antimode_log(pool, 0.75),
    "kde_log_bw1.0": antimode_log(pool, 1.0),
    "kde_log_bw1.5": antimode_log(pool, 1.5),
    "kde_linear_bw0.75": antimode_lin(pool, 0.75),
    "kde_linear_bw1.0": antimode_lin(pool, 1.0),
    "gmm_trough_R1": gmm_trough(per_round_iv[1][~per_round_iv[1].spans_stop].iai.values)[0],
    "gmm_trough_R2": gmm_trough(per_round_iv[2][~per_round_iv[2].spans_stop].iai.values)[0],
}
finite = [v for v in checks.values() if np.isfinite(v)]
lo_t, hi_t = min(finite), max(finite)
order = np.argsort(GMM.means_.ravel())
gmm_params = dict(modes_s=(10 ** GMM.means_.ravel()[order]).tolist(), weights=GMM.weights_[order].tolist(),
                  sd_log10=np.sqrt(GMM.covariances_.ravel()[order]).tolist())
kgrid = np.linspace(0.02, 8, 800)
kdens = np.exp(GMM.score_samples(np.log10(np.clip(kgrid, FLOOR, None)).reshape(-1, 1)))


def exch_summary(rnd, th):
    m = merged[rnd]
    groups, ex = segment_exchanges(m, th)
    sizes = [len(g) for g in ex]
    gaps = [m.eff[b[0]] - m.eff[a[-1]] for a, b in zip(groups[:-1], groups[1:]) if m.seg[a[-1]] == m.seg[b[0]]]
    within = [m.eff[i] - m.eff[i - 1] for g in ex for i in g[1:]]
    both = sum(1 for g in ex if set("".join(m.who[i] for i in g)) >= {"A", "B"})
    init = [m.who[g[0]] for g in ex]
    return dict(round=int(rnd), theta=th, onsets=len(m), exchanges=len(ex), in_exch=sum(sizes) / len(m),
                size_max=int(max(sizes)), both_athletes=both / len(ex),
                init_A=init.count("A") / len(ex), init_B=init.count("B") / len(ex),
                init_AB=init.count("AB") / len(ex),
                within_median=float(np.median(within)), n_within=len(within),
                gap=describe(gaps))


sens = sorted({theta, round(lo_t, 2), round(hi_t, 2)})
ex_rows = [exch_summary(r, t) for t in sens for r in (1, 2)]

spans = []
for rnd in (1, 2):
    m = merged[rnd]
    _, ex = segment_exchanges(m, theta)
    for g in ex:
        spans.append(dict(round=rnd, t0=float(m.eff[g[0]]), t1=float(m.eff[g[-1]]), size=len(g)))
pd.DataFrame(spans).to_csv(OUT / "exchange_spans.csv", index=False)

iai = pd.read_csv(OUT / "iai_series.csv")


def own_burst_share(ath, rnd, th):
    it = iai[(iai.series == ath) & (iai["round"] == rnd)].reset_index(drop=True)
    link = ((it.iai <= th) & (~it.spans_stop)).tolist()
    sizes, cur = [], 1
    for l in link:
        if l:
            cur += 1
        else:
            sizes.append(cur)
            cur = 1
    sizes.append(cur)
    return sum(s for s in sizes if s >= 2) / (len(it) + 1)


own = {f"{a}R{r}": {str(t): own_burst_share(a, r, t) for t in (theta, 1.5, 2.0, 2.5)}
       for a in "AB" for r in (1, 2)}

acts["scoring_capable"] = acts.code.map(scoring_capable)
sc, combo = {}, {}
for (rnd, ath), sub in acts.groupby(["round", "athlete"]):
    s = sub[sub.scoring_capable].sort_values("eff").reset_index(drop=True)
    gaps = [s.eff[i] - s.eff[i - 1] for i in range(1, len(s)) if s.seg[i] == s.seg[i - 1]]
    sc[f"{ath}R{rnd}"] = dict(n_actions=len(s), share=len(s) / len(sub), **describe(gaps))
    combo[f"{ath}R{rnd}"] = float((sub.combo == "Y").mean())

# cross-athlete share of the short intervals of the combined series
short_n = short_cross = 0
for rnd in (1, 2):
    m = merged[rnd]
    for i in range(1, len(m)):
        if m.seg[i] == m.seg[i - 1] and (m.eff[i] - m.eff[i - 1]) < 0.75:
            short_n += 1
            short_cross += int(m.who[i] != m.who[i - 1] or len(m.who[i]) == 2 or len(m.who[i - 1]) == 2)

res = dict(theta=theta, gmm=gmm_params, theta_checks=checks, theta_range=[lo_t, hi_t], sens_thetas=sens,
           exchanges=ex_rows, own_burst=own, scoring=sc, combo=combo, pool_n=int(len(pool)),
           short_lt_075=dict(n=short_n, cross_athlete=short_cross),
           mix_grid=kgrid.tolist(), mix_dens=kdens.tolist())
json.dump(res, open(OUT / "exchanges.json", "w"), indent=1, default=float)

print("theta", theta, gmm_params)
for k, v in checks.items():
    print(f"  {k:28s} {v:.2f}")
for r in ex_rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "gap"},
          "gap median %.2f IQR %.2f-%.2f cv %.2f n %d" % (r["gap"]["median"], r["gap"]["q1"], r["gap"]["q3"], r["gap"]["cv"], r["gap"]["n"]))
print("own burst", {k: {t: round(v, 2) for t, v in d.items()} for k, d in own.items()})
print("scoring", {k: (v["n_actions"], round(v["share"], 2), v["n"], round(v["median"], 2), round(v["cv"], 2), round(v["tempo"], 1)) for k, v in sc.items()})
print("combo", {k: round(v, 2) for k, v in combo.items()})
print("short <0.75:", short_n, "cross-athlete or involving a simultaneous onset:", short_cross)
