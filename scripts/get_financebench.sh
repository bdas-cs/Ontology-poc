#!/usr/bin/env bash
# Sparse-clone FinanceBench: only data/ (questions + doc metadata) and pdfs/.
set -euo pipefail
DEST="${1:?usage: get_financebench.sh <dest>}"

if [ -d "$DEST/repo/.git" ]; then
  git -C "$DEST/repo" pull --ff-only
else
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/patronus-ai/financebench.git "$DEST/repo"
  git -C "$DEST/repo" sparse-checkout set data pdfs
fi

echo "questions: $(wc -l < "$DEST/repo/data/financebench_open_source.jsonl")"
echo "pdfs:      $(ls "$DEST/repo/pdfs" | wc -l)"
du -sh "$DEST/repo/pdfs"
