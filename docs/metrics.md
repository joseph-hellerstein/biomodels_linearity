# Metrics for Assessing Nonlinearity

This writeup describes a new metric for quantifying nonlinearity of a timecourse.

A timecourse with duration $T$ is subdivided into a $M$ segments so that the
original function is well approximated by piecewise linear functions on each segment.
We use $s_i$ to denote the normalized size of the $i$-th segmentw. which is the original segment size divided by $T$.
Clearly, $\sum_i s_i = 1$.
The **segment curve** is a function $\psi (s)$, where $s$ is the segment size.
$\psi (s)$ is the fraction of the timecourse that has a segment size $\leq s$.
Let $\alpha$ denote the **area under the (segment) curve (AUC)**. This is calculated as $\sum_i s_i (1 - s_i)$, where $s_i (1 - s_i)$ is the contribution to $\alpha$ of the $i$-th segment.

## Properties of AUC

### Extreme values

If $M = 1$ (one segment), then AUC = 0, and if $M = N$, then AUC = 1.

### AUC increases if a segment is partitioned

Claim: Consider ${\bf s} = s_1, \cdots, s_M$.
Consider ${\bf s}^{\prime}$ such that we replace $s_m$ by $s_{m_1}, s_{m_2}$ such that $s_m = s_{m_1} + s_{m_2}$.
Then, the AUC of the first set of segments is no larger than the AUC of the second set of segments.
This claim provides a way to connect AUC with changepoints.

Proof: $s_m$ contributes $(1 - s_m)s_m$ $\alpha$, the AUC for ${\bf s}$. So,
$$\alpha^{\prime} = \alpha - (1 - s_m) s_m + (1 - s_{m_1})s_{m_1}
+ (1 - s_{m_2})s_{m_2}$$
Note that
$$
(1 - s_m) s_m =  (1 - s_m) s_{m_1} + (1 - s_m)s_{m_2}
$$
$$
(1 - s_{m_1})s_{m_1}
+ (1 - s_{m_2})s_{m_2}
$$
since $s_{m_1}, s_{m_2} < s_m$.
So, $\alpha^{\prime} > \alpha$.

### Bounds on AUC

We construct two sets of $M$ segments that will yield $\alpha_L$, the AUC lower bound, and $alpha_H$, the AUC upper bound.

For the upper bound, $s_i = \frac{1}{M}$ and so 
$$
\alpha_H = M \frac{1}{M} (1 - \frac{1}{M}) = \frac{M-1}{M}
$$
Clearly, $lim_{M \rightarrow \infty} \alpha_H =1$.

For the lower bound, let $s^*$ be the size of the smallest segment permitted for the simulation data.
We assume that as $M \rightarrow \infty$, $s^*, M s^* \rightarrow 0$. Our configuration of the segments is that
there are $M-1$ segments of size $s^*$, and one segment of size $1 - (M-1)s^*$. For large $M$,
$$
lim_{M \rightarrow \infty} \alpha_L = (M -1) s^* (1 - s^*) + (1 - (M -1) s^*)(M -1) s^*
= 0
$$

We begin with Claim 1: If we are given $M$ segments ${\bf s}$, then 
$$
\alpha_L \leq \alpha_{\bf s} \leq \alpha_{H}
$$
The proof is that for $0 \leq s_1 < s_2 \leq \frac{1}{2}$,
$s_1 (1- s_1) < s_2 (1 - s_2)$, which is clear from the concave curve. So, the sum of these terms establishes the claim.

Now we prove the lower bound. We convert ${\bf s}$ to ${\bf s}^{\prime}$ by having $s^{\prime}_i = min(s, 1-s)$. Note that $\alpha_{\bf s} = \alpha_{{\bf s}^{\prime}}$.
Also, $s_{Li} \leq s^{\prime}_i$.

For the upper bound, we use the method of LaGrange Multipliers.
$f ({\bf s}) = \sum_i s_i (1 - s_i)$ with the constraint that
$g({\bf s} ) = \sum s_i = 1$ for $0 \leq s_i \leq 1$.
The solution is obtained from the proportionality of the gradients.
That is, $\nabla f = \lambda \nabla g$. Or,
$$
1 - 2 s_i = \lambda \equiv s_i = \frac{1 - \lambda}{2}
$$
From the constraint, we know that $M \frac{1 - \lambda}{2} = 1$ and so
$\lambda = 1 - \frac{2}{M}$.
So, we conclude that $s_i = \frac{1}{M},$
which is ${\bf s}_H$.

### Collections of changepoints of the same size may have different AUC

This follow immediately from the calculation of $\alpha_L, \alpha_H$. For a large number of segments, AUC can be anywhere in the interval $(0, 1)$.
