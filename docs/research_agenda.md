# Research Agenda

## How well can BioModels be modelled by a system of linear differential equaitons?

- [x] Use ``pySindy`` to fit a coupled linear system to the model.
- [x] Score the fit of a species timecourse by accuracy, $|true - predicted|/true$, where $true > 0$.
- [x] Evaluate the density of coefficients in the Jacobian and forcing inputs.
  
  ## How robust is linearity to perturbations of initial values?

- [x] repeat the linear studies with perturbations of $\pm 5\%$, $\pm 19\%$, $\pm 20\%$, and $\pm 50\%$.

## How well are BioModels timecourses estimated by piecewise-linear approximations?

- [x] Build ``PiecewiseSystemdDiscovery`` that does piecewise estimation given changepoints.
- [x] Build approaches to automating the selection of changepoints. Considerations are efficiency of finding changepoints and effectiveness in terms of minimizing the number of changepoints to achieve a desired accuracy.

## What are the main reasons for nonlinear behavior?

- [ ] Analyze the nonlinear models to determine which species are nonlinear and how/when the Jacobian changes to look at reactions.
