"""Agreement between the two independent codings. Run after 02_exchanges.py.

Coder 1: the author, frame-stepped. Coder 2: second analyst, real-time logging.
Coder and procedure are therefore confounded.

Onsets are matched within the same athlete and round by an optimal one-to-one
assignment (Hungarian algorithm): the number of pairs within the tolerance window is
maximised first, then the summed absolute time difference is minimised.
Interval agreement is reported on the raw and the log scale, separately for
intervals within and between exchanges.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score

from common import (DATA, FRAME, OUT, actions, cv, describe, icc21, interval_table, load_events,
                    merged_onsets, round_length, segment_exchanges)

THETA = json.load(open(OUT / "exchanges.json"))["theta"]
e1 = load_events(DATA / "event_log_coder1.csv")
e2 = load_events(DATA / "event_log_coder2.csv")
a1, a2 = actions(e1).reset_index(drop=True), actions(e2).reset_index(drop=True)
res = dict(n_onsets=[len(a1), len(a2)],
           effective_s={f"R{r}": [round_length(e1, r), round_length(e2, r)] for r in (1, 2)})
BIG = 1e3


def match(tol, shift=0.0):
    """Optimal one-to-one matching; shift is added to coder 2's times before matching."""
    pairs = []
    for rnd in (1, 2):
        for ath in "AB":
            x = a1[(a1["round"] == rnd) & (a1.athlete == ath)]
            y = a2[(a2["round"] == rnd) & (a2.athlete == ath)]
            d = np.abs(x.video_time_s.values[:, None] - (y.video_time_s.values[None, :] + shift))
            cost = np.where(d <= tol + 1e-9, d, BIG)
            ri, ci = linear_sum_assignment(cost)
            pairs += [(x.index[i], y.index[j]) for i, j in zip(ri, ci) if cost[i, j] < BIG]
    return pairs


def match_greedy(tol):
    pairs = []
    for rnd in (1, 2):
        for ath in "AB":
            x = a1[(a1["round"] == rnd) & (a1.athlete == ath)]
            y = a2[(a2["round"] == rnd) & (a2.athlete == ath)]
            cand = sorted((abs(p - q), i, j) for i, p in zip(x.index, x.video_time_s)
                          for j, q in zip(y.index, y.video_time_s) if abs(p - q) <= tol + 1e-9)
            ui, uj = set(), set()
            for _, i, j in cand:
                if i in ui or j in uj:
                    continue
                ui.add(i); uj.add(j); pairs.append((i, j))
    return pairs


def primary(code):
    for ch in code.replace("u", ""):
        if ch in "CcfFSsOTHRNXPJj-":
            return {"c": "C", "F": "f", "s": "S", "j": "J"}.get(ch, ch)
    return "?"


def dt(pairs):
    return np.array([a2.video_time_s[j] - a1.video_time_s[i] for i, j in pairs])


base = match(0.5)
offset = float(np.median(dt(base)))          # coder 2 minus coder 1
res["offset_median_s"] = offset
res["greedy_0.5"] = len(match_greedy(0.5))

windows = [("0.5 s", 0.5), ("0.3 s", 0.3), ("0.2 s", 0.2), ("0.1 s", 0.1),
           ("2 frames (0.067 s)", 2 * FRAME), ("1 frame (0.033 s)", FRAME)]
tab = []
for label, tol in windows:
    p_raw, p_adj = match(tol), match(tol, shift=-offset)
    tab.append(dict(window=label, tol=tol, matched=len(p_raw), pct_c1=len(p_raw) / len(a1), pct_c2=len(p_raw) / len(a2),
                    matched_offset_removed=len(p_adj), pct_c1_offset_removed=len(p_adj) / len(a1)))
res["matching"] = tab


def agreement(pairs):
    c1 = [a1.code[i] for i, _ in pairs]
    c2 = [a2.code[j] for _, j in pairs]
    d = dt(pairs) * 1000
    return dict(n=len(pairs), kappa_full=float(cohen_kappa_score(c1, c2)),
                kappa_primary=float(cohen_kappa_score([primary(c) for c in c1], [primary(c) for c in c2])),
                raw_agreement_primary=float(np.mean([primary(a) == primary(b) for a, b in zip(c1, c2)])),
                dt_mean_ms=float(d.mean()), dt_sd_ms=float(d.std(ddof=1)),
                loa_ms=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                within_2_frames=float(np.mean(np.abs(d) <= 2 * FRAME * 1000 + 1e-6)),
                within_200ms=float(np.mean(np.abs(d) <= 200)), within_300ms=float(np.mean(np.abs(d) <= 300)))


res["agreement_0.5"] = agreement(base)
narrow = match(0.3)
res["agreement_0.3"] = agreement(narrow)
res["agreement_0.3_offset_removed"] = agreement(match(0.3, shift=-offset))


def interval_agreement(v1, v2, floor=FRAME):
    """Raw and log-scale agreement of paired intervals (s)."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    if len(v1) < 3:
        return dict(n=len(v1))
    d = (v2 - v1) * 1000
    l1 = np.log(np.clip(v1, floor, None))
    l2 = np.log(np.clip(np.abs(v2), floor, None))
    dl = l2 - l1
    return dict(n=len(v1), icc_raw=icc21(v1, v2), icc_log=icc21(l1, l2),
                mean_diff_ms=float(d.mean()), loa_ms=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                ratio_mean=float(np.exp(dl.mean())),
                ratio_loa=[float(np.exp(dl.mean() - 1.96 * dl.std(ddof=1))), float(np.exp(dl.mean() + 1.96 * dl.std(ddof=1)))],
                median_c1=float(np.median(v1)), range_c1=[float(v1.min()), float(v1.max())],
                n_floor=int(((v1 < floor) | (np.abs(v2) < floor)).sum()), n_reversed=int((v2 < 0).sum()))


def paired_intervals(pairs):
    """(i) each athlete's own intervals between consecutive matched onsets;
    (ii) combined-series intervals between consecutive matched onsets of either athlete,
    split at theta (coder 1's interval) into within- and between-exchange."""
    own1, own2 = [], []
    key = {}
    for i, j in pairs:
        key.setdefault((a1["round"][i], a1.athlete[i]), []).append((i, j))
    for lst in key.values():
        lst.sort(key=lambda p: a1.video_time_s[p[0]])
        for (i, j), (i2, j2) in zip(lst[:-1], lst[1:]):
            if a1.seg[i] != a1.seg[i2]:
                continue
            own1.append(a1.video_time_s[i2] - a1.video_time_s[i])
            own2.append(a2.video_time_s[j2] - a2.video_time_s[j])
    w1, w2, b1, b2 = [], [], [], []
    for rnd in (1, 2):
        lst = sorted([p for p in pairs if a1["round"][p[0]] == rnd],
                     key=lambda p: (a1.video_time_s[p[0]], a1.athlete[p[0]]))
        for (i, j), (i2, j2) in zip(lst[:-1], lst[1:]):
            if a1.seg[i] != a1.seg[i2]:
                continue
            v1 = a1.video_time_s[i2] - a1.video_time_s[i]
            v2 = a2.video_time_s[j2] - a2.video_time_s[j]
            (w1 if v1 <= THETA else b1).append(v1)
            (w2 if v1 <= THETA else b2).append(v2)
    return dict(own=interval_agreement(own1, own2), combined_all=interval_agreement(w1 + b1, w2 + b2),
                within_exchange=interval_agreement(w1, w2), between_exchange=interval_agreement(b1, b2))


res["intervals_0.5"] = paired_intervals(base)
res["intervals_0.3"] = paired_intervals(narrow)

# derived metrics from each coder's own series
derived = {}
for name, e, a in (("c1", e1, a1), ("c2", e2, a2)):
    d = {}
    for rnd in (1, 2):
        L = round_length(e, rnd)
        for ath in "AB":
            sub = a[(a["round"] == rnd) & (a.athlete == ath)]
            it = interval_table(sub)
            x = it[~it.spans_stop].iai.values
            t = sub.eff.values
            n1, n3 = int((t < L / 3).sum()), int((t >= 2 * L / 3).sum())
            d[f"{ath}R{rnd}"] = dict(onsets=len(sub), **{k: v for k, v in describe(x).items() if k in ("n", "median", "q1", "q3", "cv", "tempo")},
                                     thirds=[n1, len(sub) - n1 - n3, n3], D=n3 / n1)
        m = merged_onsets(a[a["round"] == rnd])
        groups, ex = segment_exchanges(m, THETA)
        it = interval_table(m)
        x = it[~it.spans_stop].iai.values
        joint = [g for g in ex if {"A", "B"} <= set("".join(m.who[i] for i in g))]
        inj = {i for g in joint for i in g}
        d[f"ABR{rnd}"] = dict(onsets=len(m), simultaneous=int((m.who == "AB").sum()),
                              **{k: v for k, v in describe(x).items() if k in ("n", "median", "cv", "tempo")},
                              exchanges=len(ex), in_exch=sum(len(g) for g in ex) / len(m),
                              unanswered={ath: int(sum(1 for i in range(len(m)) if ath in m.who[i] and i not in inj)) for ath in "AB"},
                              opened_B=sum(1 for g in joint if m.who[g[0]] == "B") / len(joint))
    derived[name] = d
res["derived"] = derived

json.dump(res, open(OUT / "interrater.json", "w"), indent=1, default=float)

print("onsets", res["n_onsets"], "effective", res["effective_s"], "median offset c2-c1 %.3f s" % offset, "greedy@0.5:", res["greedy_0.5"])
for r in tab:
    print(f"  ±{r['window']:20s} matched {r['matched']:3d} ({r['pct_c1']:.0%} of c1, {r['pct_c2']:.0%} of c2) | offset removed {r['matched_offset_removed']:3d} ({r['pct_c1_offset_removed']:.0%})")
for k in ("agreement_0.5", "agreement_0.3", "agreement_0.3_offset_removed"):
    print(k, {a: (round(b, 2) if isinstance(b, float) else [round(x) for x in b] if isinstance(b, list) else b) for a, b in res[k].items()})
for k in ("intervals_0.5", "intervals_0.3"):
    for kk, v in res[k].items():
        print(k, kk, {a: (round(b, 3) if isinstance(b, float) else [round(x, 2) for x in b] if isinstance(b, list) else b) for a, b in v.items()})
for s in ("AR1", "AR2", "BR1", "BR2", "ABR1", "ABR2"):
    print(s, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in derived["c1"][s].items()}, "\n    ",
          {k: (round(v, 2) if isinstance(v, float) else v) for k, v in derived["c2"][s].items()})
