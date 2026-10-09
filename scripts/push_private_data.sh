#!/usr/bin/env bash
# Copy Mac-only benchmark files to the droplet (they are deliberately not in git):
# the Bharat question set (no blanket licence on the source text) and its extraction set.
#   scripts/push_private_data.sh <droplet-ip> [--user root]
set -euo pipefail
HOST="${1:?usage: $0 <droplet-ip> [--user root]}"; shift
RUSER=root; [ "${1:-}" = "--user" ] && RUSER="$2"
L="${POC_DATA_DIR:-$HOME/ontology-poc-data}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
"${SSH[@]}" "$RUSER@$HOST" "mkdir -p ontology-poc-data/bharat/db"
rsync -az -e "${SSH[*]}" "$L/bharat/questions.jsonl" "$RUSER@$HOST:ontology-poc-data/bharat/"
[ -f "$L/bharat/db/gov_extract_docs.txt" ] && \
  rsync -az -e "${SSH[*]}" "$L/bharat/db/gov_extract_docs.txt" "$RUSER@$HOST:ontology-poc-data/bharat/db/"
echo "pushed Bharat questions (+ extraction set) to $HOST"
