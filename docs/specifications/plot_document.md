# Plot Document

This writeup describes how to create a latex document that properly orders plots (PDF documents) in ``plots/changepoint_timecourses`` in a latext document.

Each plot in the referenced directory has a title that specifies a value of ``AUC`` and ``p50 species p10 accuracy``. Do the following:
1. Extract the value of AUC for each plot/PDF file in the directory.
1. Construct a CSV file that has the columns: model number, AUC, p50.
1. Sort the file by increasing value of AUC, using increasing p50 accuracy as the secondary sort key
1. Construct a latex document that creates a figure for each file where the figures are ordered by their AUC.