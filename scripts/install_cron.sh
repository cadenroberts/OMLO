#!/usr/bin/env bash
#
# Install cron jobs for the ORAM experiment suite.
#
# Schedules the experiment orchestrator to run once, starting
# at the next quarter-hour. Also installs a nightly analysis
# re-run to pick up any newly completed experiment data.
#
# Usage:
#   ./scripts/install_cron.sh          # install cron entries
#   ./scripts/install_cron.sh --remove # remove cron entries

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scripts/run_experiments.sh"
LOG="${PROJECT_ROOT}/results/logs/cron_experiment.log"
ANALYSIS_LOG="${PROJECT_ROOT}/results/logs/cron_analysis.log"
CRON_TAG="# oram-thesis-experiments"

mkdir -p "$(dirname "$LOG")"

remove_entries() {
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab - 2>/dev/null || true
    echo "Removed existing ORAM cron entries."
}

if [[ "${1:-}" == "--remove" ]]; then
    remove_entries
    exit 0
fi

# Remove old entries first (idempotent)
remove_entries

# Schedule experiment run: starts in 5 minutes, runs once (lock file prevents re-run)
NEXT_MIN=$(( ($(date +%M) + 5) % 60 ))
NEXT_HOUR=$(date +%H)
# If minute wrapped, increment hour
if [ "$NEXT_MIN" -lt "$(date +%M)" ]; then
    NEXT_HOUR=$(( (NEXT_HOUR + 1) % 24 ))
fi

# Schedule nightly analysis re-run at 3:00 AM
CRON_EXPERIMENT="${NEXT_MIN} ${NEXT_HOUR} * * * ${RUNNER} >> ${LOG} 2>&1 ${CRON_TAG}"
CRON_ANALYSIS="0 3 * * * ${RUNNER} --phase 5 >> ${ANALYSIS_LOG} 2>&1 ${CRON_TAG}"

# Append to existing crontab
(crontab -l 2>/dev/null || true; echo "$CRON_EXPERIMENT"; echo "$CRON_ANALYSIS") | crontab -

echo "Cron jobs installed:"
echo "  Experiments: will start at $(printf '%02d:%02d' $NEXT_HOUR $NEXT_MIN) today"
echo "  Analysis:    nightly at 03:00"
echo ""
echo "View with:   crontab -l"
echo "Remove with: $0 --remove"
echo "Logs at:     ${LOG}"
