#!/usr/bin/env bash
# Pull GPU-droplet results back to the Mac without touching the Mac's own results.
#
#   scripts/pull_results.sh <droplet-ip> [--tag gpu] [--user root] [--with-lancedb] [--dry-run]
#
# Remote ~/ontology-poc-data/<dataset>/runs/  ->  local ~/ontology-poc-data/<dataset>/runs-<tag>/
# Remote ~/ontology-poc-data/<dataset>/db/    ->  local ~/ontology-poc-data/<dataset>/db-<tag>/
#        (pages.parquet, ontology.duckdb, extract_cache/, small files; LanceDB only with --with-lancedb)
# Remote ~/ontology-poc-data/telemetry/ + server.log -> local ~/ontology-poc-data/telemetry-<tag>/
#
# Never pulls the venv, the HF model cache or raw datasets. Safe to re-run (rsync is incremental).
# Run `make telemetry-stop` on the droplet first so telemetry.duckdb is not copied mid-write.
set -euo pipefail

HOST="" TAG="gpu" RUSER="root" LANCE=0 DRY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --user) RUSER="$2"; shift 2 ;;
    --with-lancedb) LANCE=1; shift ;;
    --dry-run|-n) DRY=(--dry-run); shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) HOST="$1"; shift ;;
  esac
done
[ -n "$HOST" ] || { echo "usage: $0 <droplet-ip> [--tag gpu] [--user root] [--with-lancedb] [--dry-run]" >&2; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--tag must be [A-Za-z0-9_-]+" >&2; exit 2; }

LOCAL="${POC_DATA_DIR:-$HOME/ontology-poc-data}"
LOCAL="${LOCAL/#\~/$HOME}"
REMOTE="ontology-poc-data"   # relative to the remote user's home
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
R="$RUSER@$HOST"

datasets=$("${SSH[@]}" "$R" "cd $REMOTE 2>/dev/null && for d in */runs; do [ -d \"\$d\" ] && echo \"\${d%/runs}\"; done" || true)
[ -n "$datasets" ] || { echo "no $REMOTE/*/runs on $R" >&2; exit 1; }

sync() { rsync -az ${DRY[@]+"${DRY[@]}"} -e "${SSH[*]}" "$@"; }

for ds in $datasets; do
  echo "== $ds"
  mkdir -p "$LOCAL/$ds/runs-$TAG" "$LOCAL/$ds/db-$TAG"
  sync "$R:$REMOTE/$ds/runs/" "$LOCAL/$ds/runs-$TAG/"
  db_ex=(--exclude 'raw/' --exclude '*.tmp')
  [ "$LANCE" = 1 ] || db_ex+=(--exclude 'lancedb/')
  sync "${db_ex[@]}" "$R:$REMOTE/$ds/db/" "$LOCAL/$ds/db-$TAG/" || echo "   (no db/ for $ds)"
done

echo "== telemetry"
mkdir -p "$LOCAL/telemetry-$TAG"
sync --exclude '*.pid' "$R:$REMOTE/telemetry/" "$LOCAL/telemetry-$TAG/" || echo "   (no telemetry/)"
sync "$R:$REMOTE/server.log" "$LOCAL/telemetry-$TAG/server.log" || echo "   (no server.log)"

echo
echo "Pulled into $LOCAL/<dataset>/runs-$TAG, db-$TAG and $LOCAL/telemetry-$TAG (Mac runs/ untouched)."
du -sh "$LOCAL"/*/runs-"$TAG" "$LOCAL/telemetry-$TAG" 2>/dev/null || true
