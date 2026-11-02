# Metrics for Assessing Nonlinearity

This writeup describes a new metric for quantifying nonlinearity of a timecourse.

A timecourse with duration $T$ is subdivided into a $M$ segments so that the
original function is well approximated by piecewise linear functions on each segment.
We use $s_i$ to denote the normalized size of the $i$-th segmentw. which is the original segment size divided by $T$.
Clearly, $\sum_i s_i = 1$.
The **segment configuration** refers to the number of segments of each size.
The **cumulative segment curve** is a function $\psi (s)$, where $s$ is the segment size, and its value is the fraction of the timecourse covered by segments of size $s$ or smaller.
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

### Extreme Segment Configurations

We find the segment configurations that minimize and maximize $\alpha$.

### Upper Bound

For the upper bound, we find a configuration that maximizes $\alpha$ if there are $M$ segments and segment sizes are in $[0, 1]$.
We use the method of LaGrange Multipliers.
$f ({\bf s}) = \sum_i s_i (1 - s_i)$ with the constraint that
$g({\bf s} ) = \sum s_i = 1$ for $0 \leq s_i \leq 1$.
The solution is obtained from the proportionality of the gradients.
That is, $\nabla f = \lambda \nabla g$. Or,
$$
1 - 2 s_i = \lambda \equiv s_i = \frac{1 - \lambda}{2}
$$
From the constraint, we know that $M \frac{1 - \lambda}{2} = 1$ and so
$\lambda = 1 - \frac{2}{M}$.
So, we conclude that $s_i = \frac{1}{M}.$
$$
\psi_H(s) = M \frac{1}{M}(1 - \frac{1}{M}) = (1 - \frac{1}{M})
$$
Note that $lim_{M \rightarrow \infty} \psi_H(s) = 1$
That is, a segment configuration that maximizes AUC has $M$ segments of equal size.

### Lower Bound
For the lower bound, we again consider $M$ segments. Here, $(M - 1)$ are small and the last one is large. That is,
$$
\psi_L(s) = \left( (M -1) s (1 - s ) + (1 - (M-1) s)(M-1)s \right)
$$
Observe that $lim_{s \rightarrow 0} \psi(s) \rightarrow 0$.
