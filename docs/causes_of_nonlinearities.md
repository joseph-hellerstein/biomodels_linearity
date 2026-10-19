# Identifying Causes of Nonlinearities

This document describes a technical approach that relates nonlinearities in models to the choice of rate laws.

## Proposed Approach

1. Find **nonlinear regions (NLR)**, a cluster of small segments in the timecourse.

2. Identify causal reactions.

   1. For each NLR
       1. Analyze the stoichiometry matrix to find for each species, the reaction that most contributes to its change. This may invovle implemeting a new class that attributes total increases and total decreases to reactions.
       2. Identify the reaction(s) that most contribute to the change. For example, if species concentration increases, then look for those reactions whose change in flux contribute to the increase (via the stoichiometry matrix). These values are the change in reaction flux times it's entry in the stoichiometry matrix for the species.

## Validating causes of nonlinearities

1. Replace the rate law with a linear rate law and assess linearity. The assessment should take place at the beginning of the NLR.