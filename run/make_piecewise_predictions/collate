#!/bin/bash
# Collates the result of make_piecewise_predictions.py

BASEFILENAME="piecewise_predictions"

cd data
cp ${BASEFILENAME}_0.csv ${BASEFILENAME}.csv
for i in {1..17}; do
    tail -n +2 ${BASEFILENAME}_${i}.csv >> ${BASEFILENAME}.csv
done