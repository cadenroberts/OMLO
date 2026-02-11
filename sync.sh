#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MSG="${1:-"sync: $(date '+%Y-%m-%d %H:%M:%S')"}"

git add -A
if git diff --cached --quiet; then
    echo "Nothing to commit."
    exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
echo "Committed and pushed to origin/$BRANCH."
