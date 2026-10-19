#!/bin/bash
# Checks status of the running python processes for calculate_linear_prediction_scores
ps aux | grep "%MEM"  > /tmp/t.header
cat /tmp/t.header
ps aux | grep "calculate_linear_prediction_scores" 
