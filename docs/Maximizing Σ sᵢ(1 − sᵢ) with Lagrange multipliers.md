# Maximizing Σ sᵢ(1 − sᵢ) with Lagrange multipliers

Oct 18, 2026 · @Joe

The maximum on the simplex is f = 1 − 1/n, at the uniform distribution sᵢ = 1/n. Because f is concave, any point satisfying the Lagrange conditions is the global maximum, so no corners need checking.

## (a) Box constraint only: 0 < sᵢ < 1

The maximum is s = (½, …, ½) with f = n/4.

Setting the gradient to zero:

```latex
\frac{\partial f}{\partial s_i} = 1 - 2s_i = 0 \;\Rightarrow\; s_i = \tfrac{1}{2}
```

This point lies strictly inside the box, so no constraint is active and all KKT multipliers are zero (αᵢ = βᵢ = 0). The Hessian −2I is negative definite, so it is a strict global maximum. Each term sᵢ(1 − sᵢ) peaks at ¼ when sᵢ = ½, and the terms are independent.

## (b) Equality constraint only: Σᵢ sᵢ = 1

The maximum is sᵢ = 1/n with f = 1 − 1/n and multiplier λ = 2/n − 1.

```latex
L = \sum_i s_i(1 - s_i) + \lambda\left(\sum_i s_i - 1\right)
```

Setting ∂L/∂sᵢ = 1 − 2sᵢ + λ = 0 makes every sᵢ equal, and the constraint fixes them:

```latex
s_i = \frac{1}{n}, \qquad \lambda = \frac{2}{n} - 1, \qquad f_{\max} = 1 - \frac{1}{n}
```

Second-order check: the Hessian of L is −2I, negative definite on the tangent space {d : Σ dᵢ = 0}. Since f is concave and the constraint is affine, the local maximum is global.

A check without calculus: on the constraint f = 1 − Σsᵢ², and Cauchy–Schwarz gives the bound below, with equality exactly when all sᵢ are equal.

```latex
\sum_i s_i^2 \ge \frac{\left(\sum_i s_i\right)^2}{n} = \frac{1}{n}
```

The multiplier is a sensitivity. Relax the constraint to Σsᵢ = c: then sᵢ = c/n and f\* = c − c²/n, so df\*/dc = 1 − 2c/n = 1 − 2/n at c = 1. That equals −λ; the sign comes from writing the constraint term as +λ(Σsᵢ − c).

## (a) and (b) together: the simplex

The maximum is the uniform distribution, sᵢ = 1/n, with f = 1 − 1/n.

The point from (b) already satisfies 0 < 1/n < 1 for n ≥ 2. So the box constraints are inactive, their multipliers are zero (μᵢ = 0), and adding the box changes nothing. In Gini-impurity terms, the uniform distribution is the most impure, or least certain, distribution.

Consistency check: for n = 2, (a) and (b) both give s = (½, ½) and f = ½. That point satisfies both constraints, so the answers must agree there.

## Contrast with the minimum

The maximum comes from tangency at an interior point; the minimum comes from the corners and the inequality multipliers.

|  | Maximum | Minimum |
| --- | --- | --- |
| Where | Interior point (sᵢ = 1/n) | Corners of the simplex |
| How it's found | Lagrange tangency: ∇f ∥ ∇h | Inequality multipliers (KKT) |
| Box multipliers | All zero | Positive |
| Uniqueness | Unique (f strictly concave) | Every vertex ties at f = 0 |
