#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/ard_template.yaml}"
DEVICE="${2:-}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
ARGS=(--config "${CONFIG}")
if [[ -n "${DEVICE}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

python -m guard.ard_train "${ARGS[@]}"
