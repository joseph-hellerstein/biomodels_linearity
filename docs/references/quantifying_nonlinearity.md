**Directly on point**
- **Philipp (2021), "The Nonlinearity Coefficient"** (arXiv 2105.12210). The author motivates measuring the degree of nonlinearity by noting that one function can be closely approximated by three line segments while another requires many. He ultimately settles on a derivative-based metric, but the segment-count intuition is discussed explicitly. [arxiv](https://arxiv.org/pdf/2105.12210)
- **Frenzen, Sasao & Butler (2010), "On the number of segments needed in a piecewise linear approximation,"** J. Comput. Appl. Math. 234(2). This work arose because high-speed circuits that implement arithmetic functions as piecewise linear approximations created a need to understand how the segment count depends on the interval and the desired error. A key result is that the count grows like ∫√|f''| dx / √ε, so that integral works as a natural nonlinearity measure. [nps](https://calhoun.nps.edu/entities/publication/19e09e58-6402-4f24-aaa9-0e463eaae0b8)
- **CaPLa (2025, bioRxiv).** This paper proposes a novel measure of piecewise-linear approximability of data, based on the "PLA-size" function. That function maps each ε to the smallest number of segments b(ε) achieving an ε-approximation. [nih](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12338547/) [nih](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12338547/)

**Neural networks (counting linear regions)**
- Montúfar et al. (2014) and Hanin & Rolnick (2019, "Complexity of Linear Regions in Deep Networks") measure the expressivity of ReLU networks by how many linear pieces they actually realize.

**Optimization/MILP (error vs. breakpoints trade-off)**
- **LinA (CIRRELT 2021-39)** computes piecewise linear functions that minimize the number of segments for a bounded pointwise error. [cirrelt](https://cirrelt.ca/documentstravail/cirrelt-2021-39.pdf)
- **arXiv 2309.10666** derives a precomputable trade-off between error and breakpoints for first-order loss functions. [arxiv](https://arxiv.org/pdf/2309.10666)

**Related but not piecewise**
- In control theory, Desoer & Wang (1980) and Allgöwer and colleagues (1990s–2000) define a nonlinearity measure as the distance to the *best single* linear approximation.
- Segmented regression (Bai–Perron breakpoint tests, Muggeo's `segmented` R package) is used in ecology and economics to test *whether* a relationship is nonlinear.

If you tell me your setting (scalar functions, data, networks, dynamical systems), I can narrow this down.

Sources:
- [The Nonlinearity Coefficient (arXiv 2105.12210)](https://arxiv.org/pdf/2105.12210)
- [Frenzen, Sasao & Butler 2010](https://calhoun.nps.edu/entities/publication/19e09e58-6402-4f24-aaa9-0e463eaae0b8)
- [Efficiency of Learned Indexes on Genome Spectra (CaPLa)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12338547/)
- [LinA, CIRRELT 2021-39](https://cirrelt.ca/documentstravail/cirrelt-2021-39.pdf)
- [arXiv 2309.10666](https://arxiv.org/pdf/2309.10666)