# TO DO

1. Verify rate_law_analyzer

# Analyses
1. Consistency of changepoint and AUC across replications
2. Relate AUC to accuracy, like distribution of AUC for large accuracies.
3. Relate model characteristics to nonlinearity as quantified by AUC.
    1. Histogram. x-axis are classification of rate laws; y-axis is fraction of models in group. The groups are 0 < AUC < 0.4 (Low AUC); 0.4 < AUC < 0.8; 0.8 < AUC < 1.0. These are colored green, yellow, red in terms of their nonlinearity. Plot as stacked bars, with a maximum height of 3. Looking for large distinctions between green and red. May need to characterize the models in each group: number of reactions, number of species.

# Tasks
1. Validate predictions with bestchangepoint
3. Analyze variability of the AUC, number changepoints, and accuracy relationship.
   
# Plots

1. Plot CAF for each model with an accuracy constraint by choosing the max_fractional_reduction that achieves that accuarcy. Display AUC, max_fractional_reduction and also plot the timecourse with changepoints.
