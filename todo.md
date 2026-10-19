# Tasks
0. NLCurve
   1. Bug in merge of curves. Index must be scaled by the maximum boundary. Redo algorithm to: (a) find new boundaries as the union of the old boundaries; (b) construct segments; (c) properly assign fractional areas.
1. 1.1 getPSDPredctionDF uses a pickle cache.
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
