#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if [[ ! -f .venv/council-ready ]]; then
    .venv/bin/python -m pip install -e .
    touch .venv/council-ready
fi
echo "Grand Council running at http://127.0.0.1:8765"
exec .venv/bin/python -m grand_council.server
