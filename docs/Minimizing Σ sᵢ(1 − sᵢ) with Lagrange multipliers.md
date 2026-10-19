# Minimizing Σ sᵢ(1 − sᵢ) with Lagrange multipliers

Oct 18, 2026 · @Joe

The minimum is f = 0, at the vertices of the simplex (one sᵢ = 1, the rest 0). Because f is concave, the minimum comes from the corners and the inequality multipliers, not from Lagrange tangency.

## Setting up

The objective and its derivatives are:

```latex
f(s) = \sum_i s_i(1 - s_i) = \sum_i s_i - \sum_i s_i^2
```

```latex
\frac{\partial f}{\partial s_i} = 1 - 2s_i, \qquad \nabla^2 f = -2I
```

The Hessian is negative definite, so f is strictly concave. Stationary points of a concave function are maxima, so the minimum is always pushed to the boundary of the allowed region. That is where the inequality (KKT) side of the Lagrange method becomes necessary.

(a) and (b) are treated separately, then together. Together they define the probability simplex, where f becomes 1 − Σsᵢ², the Gini impurity.

## (a) Box constraint only: 0 < sᵢ < 1

The infimum is 0, approached as every sᵢ → 0 or 1, but never attained on the open box.

Unconstrained stationarity gives 1 − 2sᵢ = 0, so sᵢ = ½ for every i. Because f is concave, this is the maximum, with f = n/4.

Each term sᵢ(1 − sᵢ) is a downward parabola on (0, 1): positive inside, approaching 0 at either end. So f > 0 on the open box.

On the closed box 0 ≤ sᵢ ≤ 1, the minimum is attained at the corners sᵢ ∈ {0, 1}, and KKT confirms them. Write the constraints as −sᵢ ≤ 0 and sᵢ − 1 ≤ 0, with multipliers αᵢ, βᵢ ≥ 0:

```latex
1 - 2s_i - \alpha_i + \beta_i = 0
```

- At sᵢ = 0, only the lower bound is active (βᵢ = 0), so αᵢ = 1 ≥ 0.
- At sᵢ = 1, only the upper bound is active (αᵢ = 0), so βᵢ = 1 ≥ 0.

Geometrically, the downhill direction −∇f points out through the wall in both cases.

## (b) Equality constraint only: Σᵢ sᵢ = 1

There is no minimum: f is unbounded below on the plane Σsᵢ = 1. The Lagrange point sᵢ = 1/n is the maximum.

```latex
L = \sum_i s_i(1 - s_i) + \lambda\left(\sum_i s_i - 1\right)
```

Setting ∂L/∂sᵢ = 1 − 2sᵢ + λ = 0 gives sᵢ = (1 + λ)/2, the same value for every i. The constraint then forces:

```latex
s_i = \frac{1}{n}, \qquad \lambda = \frac{2}{n} - 1, \qquad f = 1 - \frac{1}{n}
```

This is the only point where ∇f ∥ ∇h. On the constraint surface f = 1 − Σsᵢ², and the Hessian −2I is negative on the tangent space {Σ dsᵢ = 0}, so it is a maximum.

To see there is no minimum, take s₁ = t, s₂ = 1 − t, and the rest 0. Then f = 2t(1 − t), which goes to −∞ as t → ∞.

## (a) and (b) together: the probability simplex

On the simplex, every point with k equal nonzero entries satisfies KKT, but only the vertices (k = 1) are minima.

With both constraints, f = 1 − Σsᵢ². Use the closed simplex (sᵢ ≥ 0, Σsᵢ = 1); the upper bounds sᵢ ≤ 1 are then automatic. With multiplier λ for the equality and μᵢ ≥ 0 for −sᵢ ≤ 0:

```latex
1 - 2s_i + \lambda - \mu_i = 0, \qquad \mu_i s_i = 0
```

Take k coordinates equal to 1/k and the other n − k equal to 0:

- Nonzero coordinates (μᵢ = 0): 1 − 2/k + λ = 0, so λ = 2/k − 1.
- Zero coordinates: μᵢ = 1 + λ = 2/k ≥ 0.

So every such point is a KKT point, with f = 1 − 1/k. KKT is necessary but not sufficient:

| k nonzero entries | f | What it is |
| --- | --- | --- |
| k = n (uniform) | 1 − 1/n | Global maximum |
| 1 < k < n | 1 − 1/k | Maximum within its face, not a minimum |
| k = 1 (vertex) | 0 | Global minimum |

At a vertex such as (1, 0, …, 0), the multipliers are λ = 1 and μᵢ = 2 for i ≥ 2. The active constraint normals form a cone, and −∇f = (1, −1, …, −1) lies inside it, so every feasible direction moves f uphill.

## Answer

The minimum is f = 0, at the vertices of the simplex: one sᵢ = 1 and the rest 0, a pure, completely certain distribution.

- With the strict inequalities 0 < sᵢ < 1 as written, the vertices are excluded, so 0 is an infimum that is never reached.
- The Lagrange condition on the equality constraint alone finds sᵢ = 1/n, but that is the maximum of the impurity, f = 1 − 1/n.

Because f is concave, the minimum comes from the corners, where the inequality multipliers do the work, not from tangency.
