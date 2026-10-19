# Score Class

## Motivation

1. Provide a way to score the results of a prediction that is independent of how the prediction was constructed.
2. Provide for combining predictions.
3. Provide for visualizing scores and combinations of scores.

## Terminology

1. A timecourse is a sequence of values paired with timepoints. A value may be a vector that represents multiple chemical species.
2. A true timecourse contains the actual values over time.
3. A prediction timecourse is a time course of estimated values.
4. A test result is a pair of a true timecourse and a prediction timecourse.
5. A score is a comparison of the prediction with the true time course, possibly aggregated over time and species.
6. Several examples of scores are:
   1. The absolute relative error (ARE) for a scalar is defined as follows. Let x be the true value and y be the prediction. The ARE is (y-x)/x. If x is 0, then MARE is undefined. Note that ARE has no units.
   2. The mean, minimum, maximum, median, and percentile are aggregation functions applied to a collection of AREs. Aggregation can be done over time and/or over species.

## Features

1. Construct an empty score with descriptive information.
2. Add a test result.
3. Aggregate a test result by time by calculating scalar scores at each time and applying the desired aggregations such as minimum, maximum, and mean.
4. Aggregate by species in a manner analogous to time.
5. Construct a time plot of species aggregations over time.
6. Construct a bar plot of time aggregations over species.
