#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
fi

"$ROOT/.venv/bin/python" scripts/get_data.py
"$ROOT/.venv/bin/python" -m src.train
"$ROOT/.venv/bin/python" scripts/run_analysis.py
"$ROOT/.venv/bin/python" scripts/make_report.py
