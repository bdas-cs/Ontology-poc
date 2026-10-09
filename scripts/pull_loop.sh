#!/usr/bin/env bash
# Copy droplet results back to the Mac every N minutes until stopped (Ctrl-C).
# Protects a spot droplet's work: if it is reclaimed, at most one interval is lost
# beyond what gpu_run.sh can resume.
#
#   scripts/pull_loop.sh <droplet-ip> [minutes=20] [extra pull_results.sh args...]
set -uo pipefail
HOST="${1:?usage: $0 <droplet-ip> [minutes] [pull_results args]}"; shift
MIN="${1:-20}"; [ $# -gt 0 ] && shift
DIR="$(cd "$(dirname "$0")" && pwd)"
while true; do
  echo "== $(date '+%F %T') pulling from $HOST"
  "$DIR/pull_results.sh" "$HOST" "$@" || echo "   pull failed (droplet busy or gone?) — will retry"
  sleep $((MIN * 60))
done
