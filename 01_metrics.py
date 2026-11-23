"""Interval series and summary statistics for each athlete and for the dyad.

Writes out/events_effective.csv, out/iai_series.csv, out/stoppages.csv,
out/round_qc.csv, out/restart_latency.csv and out/metrics.json.
"""
import json

import numpy as np
import pandas as pd

from common import (OUT, actions, boot_ci, cv, describe, interval_table, load_events,
                    merged_onsets, round_length, stoppages)

ev = load_events()
acts = actions(ev)
stops = stoppages(ev)

qc = []
for rnd, g in ev.groupby("round"):
    st = stops[stops["round"] == rnd]
    qc.append(dict(round=int(rnd), effective_s=round_length(ev, rnd), n_stops=len(st),
                   stop_s=round(float(st.dur.sum()), 2),
                   score_A=float(g.pts_A.fillna(0).sum()), score_B=float(g.pts_B.fillna(0).sum())))
qc = pd.DataFrame(qc)

tables, series = [], {}
for (rnd, ath), sub in acts.groupby(["round", "athlete"]):
    it = interval_table(sub)
    it["round"], it["series"] = rnd, ath
    tables.append(it)
    x = it[~it.spans_stop].iai.values
    d = describe(x)
    d.update(onsets=len(sub), n_spanning=int(it.spans_stop.sum()),
             cv_ci=boot_ci(x, cv), median_ci=boot_ci(x, np.median),
             incl=describe(it.iai.values),
             rate_per_min_effective=60 * len(sub) / round_length(ev, rnd),
             combo_prop=float((sub.combo == "Y").mean()))
    series[f"{ath}R{rnd}"] = d

for rnd, sub in acts.groupby("round"):
    m = merged_onsets(sub)
    it = interval_table(m)
    it["round"], it["series"] = rnd, "AB"
    tables.append(it)
    x = it[~it.spans_stop].iai.values
    d = describe(x)
    d.update(onsets=len(m), n_simultaneous=int((m.who == "AB").sum()),
             n_spanning=int(it.spans_stop.sum()), cv_ci=boot_ci(x, cv),
             median_ci=boot_ci(x, np.median), incl=describe(it.iai.values),
             rate_per_min_effective=60 * len(m) / round_length(ev, rnd))
    series[f"ABR{rnd}"] = d

iai = pd.concat(tables, ignore_index=True)

# restart latency: Kye-sok -> next action of either athlete
lat = []
for rnd, g in ev.groupby("round"):
    g = g.sort_values("round_time_s", kind="stable").reset_index(drop=True)
    for i, r in g[g.event == "Stop end"].iterrows():
        nxt = g[(g.index > i) & (g.event == "Action")]
        if len(nxt):
            lat.append(dict(round=int(rnd), latency=round(float(nxt.iloc[0].round_time_s - r.round_time_s), 2),
                            first=nxt.iloc[0].athlete))
lat = pd.DataFrame(lat)

# A's Round 2 CV without its two longest gaps (sensitivity quoted in the text)
a2 = iai[(iai.series == "A") & (iai["round"] == 2) & (~iai.spans_stop)].iai.values
a2_trim = np.sort(a2)[:-2]

res = dict(qc=qc.to_dict("records"), series=series,
           restart_latency=dict(median=float(lat.latency.median()), min=float(lat.latency.min()),
                                max=float(lat.latency.max()), n=len(lat)),
           A_R2_two_longest=[float(v) for v in np.sort(a2)[-2:]], A_R2_cv_without=float(cv(a2_trim)))
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)

ev.to_csv(OUT / "events_effective.csv", index=False)
iai.to_csv(OUT / "iai_series.csv", index=False)
stops.to_csv(OUT / "stoppages.csv", index=False)
qc.to_csv(OUT / "round_qc.csv", index=False)
lat.to_csv(OUT / "restart_latency.csv", index=False)

pd.set_option("display.width", 220)
print(qc.to_string())
print(f"{'series':7s} {'on':>3s} {'n':>3s} {'median':>6s} {'IQR':>11s} {'range':>12s} {'CV':>5s} {'CV 95% CI':>12s} {'tempo':>5s}")
for k, d in series.items():
    print(f"{k:7s} {d['onsets']:3d} {d['n']:3d} {d['median']:6.2f} {d['q1']:5.2f}-{d['q3']:5.2f} "
          f"{d['min']:5.2f}-{d['max']:5.2f} {d['cv']:5.2f} {d['cv_ci'][0]:5.2f}-{d['cv_ci'][1]:5.2f} {d['tempo']:5.1f}"
          f"   incl: median {d['incl']['median']:.2f} cv {d['incl']['cv']:.2f}")
print("restart latency", res["restart_latency"])
print("A R2 longest", res["A_R2_two_longest"], "cv without", round(res["A_R2_cv_without"], 2))
