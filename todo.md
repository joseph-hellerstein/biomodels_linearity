# TO DO

1. Many models show graphs that seem at odds with the accuracy displayed for model p50.  Examples are 904, 875, and 750. Debug this.
2. PSDPredictionAnalyzer.best_model_ser doesn't look correct from plot in @psd_prediction_analysis since model 5 has a strange number of changepoints for max_fractional_reduction = 0.05 (manycp=True).

# Analyses
1. Consistency of changepoint and AUC across replications
2. Relate AUC to accuracy, like distribution of AUC for large accuracies.

# Tasks
1. Validate predictions with bestchangepoint
2. PiecewiseAnalyzer
   1. Plot CDF frac changepoints for man max_fractional_reduction (MFR)
   2. Plot a single model with changepoints for different MFR
   3. Plot CAD for MFR, singular, groups
   4. getChangepointsFromFile
      1. Preparation: file selects "model" from all PWPrediction files
      2. In PSD: create a DF of this file; check parameters to match prediction; return changepoints if match, else None.
3. Analyze variability of the AUC, number changepoints, and accuracy relationship.
4. Piecewise Network Discovery
   1. _getChangepointsFromFile
      1. Optionally specify a minimum accuracy (and so ignore max_fractional_reduction). Min num_changepoints subject to constraint on accuracy.
   
# Plots

1. Plot CAF for each model with an accuracy constraint by choosing the max_fractional_reduction that achieves that accuarcy. Display AUC, max_fractional_reduction and also plot the timecourse with changepoints.
