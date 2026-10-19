#!/usr/bin/env bash
# Run make_biomodels_timecourse.py in parallel across N workers.
# Each worker receives an evenly divided slice of the 1078 BioModels.
#
# Usage: bash scripts/run_parallel_timecourse.sh <number_of_processes>

#set -euo pipefail

if [ $# -ne 1 ] || ! [[ $1 =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 <number_of_processes>" >&2
    exit 1
fi

NUM_PROCESSES=$1
TOTAL_MODELS=1078
TOTAL_MODELS=20
FIRST_MODEL=1
LAST_MODEL=$(( FIRST_MODEL + TOTAL_MODELS - 1 ))

# Ceiling division: chunk = ceil(TOTAL / N)
CHUNK=$(( (TOTAL_MODELS + NUM_PROCESSES - 1) / NUM_PROCESSES ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/activate.sh"
mkdir -p "$PROJECT_DIR/logs"

echo "Launching $NUM_PROCESSES workers (~$CHUNK models each)..."

for i in $(seq 0 $(( NUM_PROCESSES - 1 ))); do
    FIRST=$(( FIRST_MODEL + i * CHUNK ))
    LAST=$(( FIRST + CHUNK - 1 ))
    if [ "$LAST" -gt "$LAST_MODEL" ]; then
        LAST=$LAST_MODEL
    fi
    LOG="$PROJECT_DIR/logs/timecourse_${FIRST}_${LAST}.log"
    echo "  Worker $(( i + 1 )): models $FIRST–$LAST  →  $LOG"
    python "$SCRIPT_DIR/make_biomodels_timecourse.py" \
        --first_model_num "$FIRST" \
        --last_model_num  "$LAST"  \
        > "$LOG" 2>&1 &
done

echo "All workers launched.  Waiting for completion..."
wait
echo "Done."
