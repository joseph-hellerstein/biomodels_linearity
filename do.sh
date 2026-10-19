#!/bin/bash
# Runs a piecewise prediction. Argument is the max_fractional_reduction in accuracy
run/make_piecewise_predictions/make --initialize --coefficient_threshold 0.001 --max_reduction $1  --changepoint_removal
