#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then python=.venv/bin/python
elif [[ -x ../../work/demand-venv/bin/python ]]; then python=../../work/demand-venv/bin/python
else echo 'README에 따라 가상환경과 패키지를 먼저 설치하세요.'; exit 1
fi
export STOREOPS_DEMO=1
exec "$python" -m uvicorn storeops.api:app --host 127.0.0.1 --port "${PORT:-8765}"
