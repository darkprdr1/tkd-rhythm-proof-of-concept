#!/usr/bin/env bash
# Reproduces every output in out/ and fig/ from the event logs in data/.
set -euo pipefail
cd "$(dirname "$0")"
python3 01_metrics.py
python3 02_exchanges.py
python3 03_dyad_drift_censoring.py
python3 04_interrater.py
python3 05_figure1.py
