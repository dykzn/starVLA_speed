#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi

exec /usr/bin/python3 /data3/dengyongkang/my_project/starVLA/examples/LIBERO/eval_files/monitor_baseline_continuation_gpu12.py "$1"
