
# Piecewise Affine / Piecewise Linear Differential Systems

The precise mathematical setting you're describing is called **piecewise affine (PWA)** or **piecewise linear (PWL) differential equations**, a rich field with the following key references:

---

## Foundational Papers

**1. Glass & Kauffman (1973) — "The Logical Analysis of Continuous, Non-Linear Biochemical Control Networks"**
*Journal of Theoretical Biology, 39, 103–129*

This is the foundational paper for the class of piecewise affine differential (PWA) models that has since been widely used for the modelling and analysis of biological switch-like systems, such as genetic and neural networks. The state space is partitioned into rectangular regions (a "box" partition), and within each region the dynamics are a simple linear ODE with a constant target (an affine system). Solutions in each box are exponential approach-to-target, but the global trajectory is nonlinear and can exhibit limit cycles and chaos.

---

**2. Gouzé & Sari (2002/2003) — "A Class of Piecewise Linear Differential Equations Arising in Biological Models"**
*Dynamical Systems, 17(4), 299–316*

This paper is frequently cited as providing the rigorous mathematical foundation for the Glass–Kauffman class of models, including careful handling of solutions on the switching surfaces (the boundaries between regions) using Filippov's theory of differential equations with discontinuous right-hand sides.

---

**3. Casey, de Jong & Gouzé (2006) — "Piecewise-Linear Models of Genetic Regulatory Networks: Equilibria and Their Stability"**
*Journal of Mathematical Biology, 52(1), 27–56*

This paper develops the piecewise-linear formalism originally due to Glass and Kauffman into a well-suited tool for modelling genetic regulatory networks, rigorously analyzing the equilibria (including singular stationary points on switching surfaces) and their stability. This is the most commonly cited modern reference for this exact setting.

---

**4. Snoussi (1989) — "Qualitative Dynamics of Piecewise-Linear Differential Equations: A Discrete Mapping Approach"**
*Dynamics and Stability of Systems, 4(3–4), 189–207*

This paper develops a discrete mapping approach to the qualitative dynamics of piecewise-linear differential equation systems, constructing a return map from the transitions of trajectories between switching surfaces — a key analytical tool for studying periodic orbits.

---

## Detailed Analysis of Dynamics & Periodic Solutions

**5. Mestl, Plahte & Omholt (1995) — "Periodic Solutions in Systems of Piecewise-Linear Differential Equations"**
*Dynamics and Stability of Systems, 10, 179–193*

This paper analyzes periodic solutions (limit cycles) in piecewise linear differential equation systems, exploiting the fact that within each linear region the solution is analytic (matrix exponential), and periodicity conditions can be stated as algebraic matching conditions at the switching boundaries.

---

**6. "Links Between Topology of the Transition Graph and Limit Cycles in a Two-Dimensional Piecewise Affine Biological Model"**
*Journal of Mathematical Biology (2013)*

This paper studies a class of piecewise affine differential models and links the combinatorial structure of how regions connect (the transition graph) to the existence of limit cycles in the global nonlinear trajectory.

---

**7. "Probabilistic Approach for Predicting Periodic Orbits in Piecewise Affine Differential Models"**
*Bulletin of Mathematical Biology (2012)*

This paper develops a probabilistic method for predicting periodic orbits in piecewise affine differential models, working within the Glass–Kauffman framework where each region has a linear ODE and periodic trajectories arise from the nonlinear patching of solutions across region boundaries.

---

## Biological Application with Full Methodology

**8. Ochab & Puszynski (2020) — "Piece-Wise Linear Models of Biological Systems"**
*PLOS ONE, doi:10.1371/journal.pone.0243823*

This paper explicitly proposes using a linear system with switching methodology for complex biological systems, and provides a detailed methodology including analytical determination of equilibrium points, finding an analytical solution within each piece, and stability and bifurcation analysis — applied to the p53 signaling pathway as a case study comparing results with standard nonlinear ODE models. This is particularly relevant as it directly demonstrates that each piece has an analytical (linear ODE) solution.

---

## Reviews & Broader Context

**9. de Jong et al. (2004) — "Qualitative Simulation of Genetic Regulatory Networks Using Piecewise-Linear Models"**
*Bulletin of Mathematical Biology, 66(2), 301–340*

This paper provides qualitative simulation of genetic regulatory networks using piecewise-linear models, with careful treatment of solution trajectories crossing switching surfaces. It is a major review-style paper implementing the framework computationally.

**10. Gouzé & Chaves (2010) — "Piecewise Affine Models of Regulatory Genetic Networks: Review and Probabilistic Interpretation"**
*Lecture Notes in Control and Information Sciences, vol. 470, Springer*

This review covers the piecewise affine modeling framework for regulatory networks with a probabilistic interpretation for predicting which attractors (equilibria or limit cycles) are reached from given initial conditions.

---

## Key Mathematical Background

**Filippov (1988) — *Differential Equations with Discontinuous Righthand Sides*, Kluwer**

Filippov's classical book provides the mathematical foundation for rigorously defining and solving differential equations whose right-hand side is discontinuous — essential for piecewise systems where matching conditions at boundaries must be carefully handled.

---

## Summary of the Structure

In all these references, the common architecture is:

- The state space is partitioned into **regions** (boxes, polytopes, or half-spaces) by **switching thresholds**
- Within each region, **ẋ = Aᵢx + bᵢ** — a linear (affine) ODE with analytic solutions (matrix exponentials)
- The **global trajectory is nonlinear** because the matrices Aᵢ change at boundaries
- Periodic orbits, chaos, and complex attractors can all emerge from this structure

Would you like me to look further into any particular application area (control systems, circuits, biology) or a specific mathematical aspect such as stability analysis or the Filippov boundary conditions?