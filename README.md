# Measuring around rhythm — proof-of-concept data and analysis code

Data and code for the proof-of-concept analysis in:

> Sun A-C. Measuring around rhythm: a perspective on the interval structure of technical actions in taekwondo and
> combat-sport performance analysis. *International Journal of Sports Science & Coaching* (manuscript SPO-26-1694).

Everything reported in Section 5.6, Table 3 and Figure 1 of the article, and in Section S3 of its Supplementary
Material, is produced by the five scripts in this folder from the two event logs in `data/`.

## Contents

| Path | What it is |
|---|---|
| `data/event_log_coder1.csv` | Event log of the author's coding (frame-stepped; the coding reported in the article) |
| `data/event_log_coder2.csv` | Event log of the independent second analyst (logged in real time) |
| `common.py` | Loading, effective time, segments, interval tables, exchange segmentation, ICC |
| `01_metrics.py` | Interval series and summary statistics for each athlete and for the dyad; bootstrap confidence intervals; restart latency |
| `02_exchanges.py` | Provisional exchange threshold (mixture model and alternative estimates), exchange segmentation, own-burst shares, scoring-capable subseries |
| `03_dyad_drift_censoring.py` | Dyadic dependence and conditional individual components; within-round drift with exact confidence intervals and a log-linear Poisson trend; Kaplan–Meier estimates with interrupted intervals as censored |
| `04_interrater.py` | Agreement between the two codings: optimal one-to-one matching at several windows, kappa, limits of agreement, raw- and log-scale ICC within and between exchanges |
| `05_figure1.py` | Figure 1 (PDF, PNG and 300 dpi TIFF) |
| `run_all.sh` | Runs the five scripts in order |
| `out/` | Outputs written by the scripts (CSV and JSON); included so that results can be inspected without running anything |
| `fig/` | Figure 1 as produced by `05_figure1.py` |

## Running

```bash
pip install -r requirements.txt
bash run_all.sh
```

Python 3.10 or later. The analysis was run with numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9 and matplotlib 3.11.
Random seeds are fixed (bootstrap, mixture model, figure jitter), so the outputs are reproducible. Figure 1 uses the
Liberation Sans font; matplotlib substitutes its default font if it is not installed.

## The event logs

One elite international taekwondo match in a senior women's category: two athletes (A and B), two rounds. The match was
coded from broadcast video (30 frames per second) with the coding manual in Section S2 of the Supplementary Material.
The competition, date, weight category and the athletes' identities are withheld.

Columns: `round`; `video_time_s` (time in the source video file); `round_time_s` (time since that coder's Shi-jak for
the round, stoppages included); `event`; `athlete`; `code`; `combo` (Y = combination beat); `pts_A`, `pts_B` (points
recorded on the scoring athlete's row; for a Gam-jeom the point goes to the opponent); `reason`; `notes`.

Event types: `Round start` (Shi-jak), `Round end` (Keu-man), `Stop start` (Kal-yeo), `Stop end` (Kye-sok), `Action`
(one beat, stamped at the first frame of initiation), `Continuation` (a technique launched before the previous kicking
foot has landed; belongs to the previous beat and enters no interval), `Marker` (G = Gam-jeom on the athlete of that
row). Referee commands were timestamped at the first frame of the referee's hand signal, with the audio used to confirm
which command was given.

## Definitions used by the code

- **Effective time**: round time minus all completed stoppages before the event. It is defined by the referee's
  commands, not by the official clock, and exceeds 120 s in both rounds.
- **Interval**: difference in effective time between successive onsets of a series within the same stretch of
  uninterrupted play. An interval with a stoppage between its two onsets is *stoppage-spanning*; it is excluded in the
  main tables, included in a sensitivity analysis, and treated as right-censored in the Kaplan–Meier analysis.
- **Combined (dyadic) series**: both athletes' onsets in time order, onsets with identical timestamps merged.
- **Exchange**: a maximal run of at least two onsets of the combined series whose successive intervals are all ≤ θ, with
  no stoppage between them. θ is the trough of a two-component Gaussian mixture fitted to the log10 combined intervals
  of both rounds. It is **provisional**: it is derived from the data it is then used to describe.
- **Within-round drift, D**: onset count in the final third of effective time divided by the count in the first third,
  with an exact (conditional binomial) confidence interval; and the end-to-start rate ratio of an inhomogeneous Poisson
  process with log-linear rate fitted to all onset times of the round.
- **Tempo with censoring**: completed intervals divided by total time at risk (completed intervals plus waiting times up
  to an interruption), per minute.

## Caveats

This is a single match and a feasibility demonstration. The two codings differ in coder and in procedure (frame-stepped
versus real-time), which are therefore confounded. No contrast in these data is offered as a finding about either athlete.

## Licence and citation

Code: MIT Licence. Data: CC BY 4.0. Please cite the article above.

## Use of generative AI

The analysis and figure code were written with the assistance of generative AI (Claude, Anthropic) under the author's
direction; the author reviewed the code and takes responsibility for it and for all results.
