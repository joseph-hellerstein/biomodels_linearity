#!/bin/bash
# Launch calculate_linear_prediction_scores for each of the input processes
# Should be run from the root of the repository, e.g. with `./scripts/calculate_linear_prediction_scores.sh 4 --initialize`

# Default flag value
INITIALIZE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --initialize)
            INITIALIZE=true
            shift
            ;;
        *)
            if [[ -z "$N" ]]; then
                N="$1"
            else
                echo "Usage: $0 <integer> [--initialize]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check that the integer argument was provided
if [[ -z "$N" ]]; then
    echo "Usage: $0 <integer> [--initialize]"
    exit 1
fi

# Validate that the argument is a positive integer
if ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be a non-negative integer."
    exit 1
fi

# Launch p.py for each index from 1 to N
for (( i=1; i<=N; i++ )); do
    # If --initialize is set, remove the data file for this process
    if [[ "$INITIALIZE" == true ]]; then
        FILE="data/linear_predictor_scores_${i}"
        if [[ -f "$FILE" ]]; then
            mv "$FILE" /tmp
            echo "Removed $FILE"
        fi
    fi

    #(sleep 1; echo "Launching process $i of $N") &
    #(source activate.sh && python scripts/calculate_linear_prediction_scores.py "$N" "$i")
    (source activate.sh && python scripts/calculate_linear_prediction_scores.py "$N" "$i") > "err_${i}" 2>&1 &
done
