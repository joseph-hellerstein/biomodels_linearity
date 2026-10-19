"""
Rank-split SINDy for systems with LINEAR constraints.

Data assumption: you are given state trajectories X and their
derivatives Xdot directly (from simulation output, sensors, or a
numerical differentiation step you've already done upstream). You
are NOT given the Jacobian analytically - it must be estimated from
(X, Xdot) pairs.

Use case: the estimated Jacobian is (locally) rank-deficient because
of algebraic constraints, and those constraints are known to be
linear (e.g. Cx = const, a fixed linear combination of states).
This means the null space is FIXED for all time, so we can:

  1. Estimate the Jacobian AND a constant bias term from (X, Xdot)
     via least squares regression on the affine model
     Xdot ≈ X @ J.T + b (this is the ONLY way we get J and b - there
     is no analytic Jacobian available in this problem).
  2. SVD the estimated Jacobian -> split state space into a dynamic
     subspace (rank r, genuinely evolving) and a null/constraint
     subspace (dim n-r, algebraically fixed).
  3. Verify the constraint coordinates are actually constant in time
     (this is the diagnostic that confirms "linear constraint" is a
     valid assumption for this data).
  4. Fit a linear ODE with a bias term (ydot = A y + b, or SINDy with
     a linear/degree-1 library including bias, which is the same
     thing) only on the dynamic subspace coordinates.
  5. Reconstruct the full state from the dynamic fit plus the fixed
     constraint values - no functional model needed for the null
     space, since it's just a constant.

Since the Jacobian is estimated (not exact), the rank-detection
threshold `tol` in rank_split() matters more here than it would with
an analytic Jacobian - see the note in that function.

Requires: numpy, scipy, pysindy (pip install pysindy)
"""

import numpy as np
from scipy.linalg import svd
import pysindy as ps


# ----------------------------------------------------------------------
# 1. Estimate the Jacobian from data. This is the ONLY source of J in
#    this pipeline - you're given (X, Xdot) pairs, not an analytic
#    Jacobian, so everything downstream depends on this estimate
#    being reasonable.
# ----------------------------------------------------------------------
def estimate_jacobian(X, Xdot, reg=1e-8, method="ridge"):
    """
    Local-affine estimate of the Jacobian AND bias term from state
    data X (n_samples x n_states) and derivatives Xdot (n_samples x
    n_states), by solving Xdot ≈ X @ J.T + b in a least-squares sense.

    Returns (J, b):
      J : (n_states x n_states) Jacobian
      b : (n_states,) constant/bias term

    method="ridge" (default): ordinary ridge regression on the
        augmented design matrix [X | 1]. Good when noise is mostly
        in Xdot.

    method="tls": total least squares (errors-in-variables). Use this
        if X itself carries significant noise too - ridge regression
        is biased in that case. Note: TLS doesn't naturally separate
        out a bias term the way ridge does, so here we center X and
        Xdot first (removing the bias as a pre-processing step), do
        TLS on the centered data for J, then recover b from the
        centering means.
    """
    n_samples, n_states = X.shape

    if method == "ridge":
        # augment X with a column of ones to fit the bias jointly
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        reg_matrix = reg * np.eye(n_states + 1)
        reg_matrix[-1, -1] = 0.0  # don't penalize the bias term
        XtX = X_aug.T @ X_aug + reg_matrix
        XtY = X_aug.T @ Xdot
        JB = np.linalg.solve(XtX, XtY).T  # (n_states, n_states+1)
        J = JB[:, :n_states]
        b = JB[:, n_states]
        return J, b

    elif method == "tls":
        # remove the bias by centering, do TLS on the centered data
        # for J, then recover b from the means: b = xdot_mean - J @ x_mean
        x_mean = X.mean(axis=0)
        xdot_mean = Xdot.mean(axis=0)
        Xc = X - x_mean
        Xdotc = Xdot - xdot_mean

        Z = np.hstack([Xc, Xdotc])
        _, _, Vt = svd(Z, full_matrices=False)
        V = Vt.T
        Vxx = V[:n_states, :n_states]
        Vyx = V[n_states:, :n_states]
        J = -(Vyx @ np.linalg.pinv(Vxx)).T
        J = J.T
        b = xdot_mean - J @ x_mean
        return J, b

    else:
        raise ValueError(f"Unknown method: {method}")


# ----------------------------------------------------------------------
# 2. SVD-based rank split
# ----------------------------------------------------------------------
def rank_split(J, tol=1e-6):
    """
    Given an (estimated) Jacobian J (n x n), return:
      V_dyn   : (n x r) basis for the dynamic subspace
      V_null  : (n x (n-r)) basis for the constraint/null subspace
      r       : detected rank
      s       : singular values (for diagnostics / choosing tol)

    Since J here comes from regression on noisy data rather than an
    exact analytic source, the "zero" singular values will be small
    but nonzero. Look at `s` first (see __main__) and set `tol`
    relative to the gap between the last "large" and first "small"
    singular value, rather than assuming a specific numeric cutoff
    works across different datasets/noise levels.
    """
    U, s, Vt = svd(J)
    r = int(np.sum(s > tol * s[0]))  # relative threshold
    V = Vt.T
    V_dyn = V[:, :r]
    V_null = V[:, r:]
    return V_dyn, V_null, r, s


# ----------------------------------------------------------------------
# 3. Project data into dynamic / null coordinates
# ----------------------------------------------------------------------
def project_data(X, Xdot, V_dyn, V_null):
    Y_dyn = X @ V_dyn          # dynamic coordinates
    Ydot_dyn = Xdot @ V_dyn    # their derivatives
    Y_null = X @ V_null        # constraint coordinates (should be ~static
                                # or algebraically slaved)
    return Y_dyn, Ydot_dyn, Y_null


# ----------------------------------------------------------------------
# 4. Fit a LINEAR model on the dynamic subspace
#    (degree-1 SINDy library == fitting ydot = A y, a linear ODE)
# ----------------------------------------------------------------------
def fit_linear_dynamics(Y_dyn, Ydot_dyn, t=None, threshold=0.0):
    """
    Fits ydot = A y + b (affine, i.e. linear-with-bias) on the
    reduced (rank-r) coordinates.

    threshold=0.0 gives plain least squares (dense A, b). Set
    threshold > 0 if you want STLSQ's sparsity to prune small/
    spurious coefficients.
    """
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=1, include_bias=True),
        optimizer=ps.STLSQ(threshold=threshold),
    )
    if t is not None:
        model.fit(Y_dyn, t=t, x_dot=Ydot_dyn)
    else:
        model.fit(Y_dyn, x_dot=Ydot_dyn)
    return model


def extract_linear_matrix(model, r):
    """
    Pulls the fitted r x r matrix A and bias vector b out of the
    SINDy model, so you can use them directly (e.g. for eigenvalue/
    fixed-point analysis, or plugging back into a linear-system
    solver) instead of going through pysindy's simulate() every time.

    With include_bias=True, pysindy's PolynomialLibrary puts the
    bias (constant "1") feature first, followed by the r linear
    features - hence coefs[:, 0] is b and coefs[:, 1:r+1] is A.
    """
    coefs = model.coefficients()  # shape (r, 1 + r) with bias first
    b = coefs[:, 0]
    A = coefs[:, 1:r + 1]
    return A, b


# ----------------------------------------------------------------------
# 5. Check the constraint subspace is actually constant, and record
#    its value. With linear constraints, Y_null should not depend on
#    time or on Y_dyn at all - it's just a fixed constant vector.
# ----------------------------------------------------------------------
def check_and_extract_constraints(Y_null, rel_tol=1e-2):
    """
    Verifies Y_null is (numerically) constant across all samples, and
    returns that constant value.

    Raises a warning (not an error) if the spread is larger than
    rel_tol relative to the mean magnitude - that's your signal that
    the "linear constraint" assumption may not hold for this data/
    window, and you may need the more general nonlinear-constraint
    version instead.
    """
    mean_val = Y_null.mean(axis=0)
    std_val = Y_null.std(axis=0)
    scale = np.maximum(np.abs(mean_val), 1e-12)
    rel_spread = std_val / scale

    if np.any(rel_spread > rel_tol):
        print(f"WARNING: null-space coordinates are not constant "
              f"(relative spread {rel_spread}). Linear-constraint "
              f"assumption may not hold for this data.")

    return mean_val


# ----------------------------------------------------------------------
# 6. Reconstruct full-state trajectory
#    (null-space part is now a fixed constant, not per-sample data)
# ----------------------------------------------------------------------
def reconstruct_full_state(Y_dyn, null_const, V_dyn, V_null):
    n_samples = Y_dyn.shape[0]
    Y_null_repeated = np.tile(null_const, (n_samples, 1))
    return Y_dyn @ V_dyn.T + Y_null_repeated @ V_null.T


# ----------------------------------------------------------------------
# Example usage / end-to-end pipeline
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Replace this block with your actual data ---
    # You are given these two arrays directly (from simulation,
    # sensors, experiment, etc.) - there is no Jacobian to read off,
    # it's estimated below.
    #   X:    (n_samples, n_states) state trajectory
    #   Xdot: (n_samples, n_states) derivatives, already computed
    #         upstream (not finite-differenced here)
    #   t:    (n_samples,) time vector, only needed if your samples
    #         aren't already ordered/evenly spaced for pysindy's fit
    rng = np.random.default_rng(0)
    n_samples, n_states = 500, 4
    t = np.linspace(0, 10, n_samples)

    # toy example: state 3 is a fixed linear combination of states 0,1
    # (a linear constraint), so the true Jacobian is rank-deficient by
    # 1. The dynamics also carry a constant bias term, per your data.
    # Xdot here stands in for whatever derivative data you already
    # have - normally you would NOT compute this from X yourself.
    X = rng.normal(size=(n_samples, n_states))
    X[:, 3] = 2 * X[:, 0] - 0.5 * X[:, 1]
    Xdot = np.gradient(X, t, axis=0) + 0.3  # <-- stand-in only; use your real Xdot
    # --- end replace ---

    J_est, b_est = estimate_jacobian(X, Xdot, method="ridge")

    # Inspect the singular value spectrum BEFORE picking tol - with an
    # estimated Jacobian you're looking for a gap, not an exact zero.
    _, s_diag, _ = svd(J_est)
    print(f"Singular values of estimated Jacobian: {s_diag}")

    V_dyn, V_null, r, s = rank_split(J_est, tol=1e-2)
    print(f"Detected rank: {r} / {n_states}")

    Y_dyn, Ydot_dyn, Y_null = project_data(X, Xdot, V_dyn, V_null)

    # fit affine (linear + bias) dynamics on the reduced coordinates
    model = fit_linear_dynamics(Y_dyn, Ydot_dyn, t=t)
    model.print()
    A, b = extract_linear_matrix(model, r)
    print(f"Fitted linear system matrix A ({r}x{r}):\n{A}")
    print(f"Fitted bias term b: {b}")
    eigvals = np.linalg.eigvals(A)
    print(f"Eigenvalues of A (stability check): {eigvals}")

    # verify + extract the fixed constraint value
    null_const = check_and_extract_constraints(Y_null)
    print(f"Constraint (null-space) constant: {null_const}")

    X_reconstructed = reconstruct_full_state(Y_dyn, null_const, V_dyn, V_null)
    err = np.linalg.norm(X_reconstructed - X) / np.linalg.norm(X)
    print(f"Reconstruction relative error: {err:.2e}")
