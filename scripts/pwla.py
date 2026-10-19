"""
Trajectory Piecewise-Linear (TPWL) Model
=========================================

Reference:
    Rewienski & White, "A trajectory piecewise-linear approach to model
    order reduction and fast simulation of nonlinear circuits and
    micromachined devices," IEEE Trans. CAD, 22(2):155-170, 2003.

Algorithm overview
------------------
Given a nonlinear ODE:
    dx/dt = f(x)

TPWL approximates it by:

  1. TRAINING  — Simulate the full nonlinear system along a representative
     trajectory x*(t).  At selected "expansion points" s_i along that
     trajectory, linearise f around x*(t_i):

         f(x) ≈ A_i (x - s_i) + f(s_i)
               = A_i x + b_i          where b_i = f(s_i) - A_i s_i

     A_i = df/dx|_{x=s_i}  is the Jacobian at s_i.

  2. POINT SELECTION  — Use a distance-based greedy criterion: add a new
     expansion point only when the current state is "far enough" from all
     existing points (measured in Euclidean distance or a weighted norm).

  3. ONLINE EVALUATION  — At a new state x, blend the local linear models
     with weights w_i(x) that decrease with distance from s_i:

         f_TPWL(x) = sum_i  w_i(x) * [A_i x + b_i]

     where  w_i(x) = exp(-alpha * ||x - s_i||^2) / Z(x)   (Gaussian weights).

     Alternatively the nearest-neighbour (hard) weight is used: only the
     closest expansion point contributes (w_i* = 1, all others 0).

This file implements both hard and soft weighting and demonstrates the method
on two nonlinear ODE systems of increasing complexity.
"""

from __future__ import annotations

import numpy as np # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from scipy.linalg import expm # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from typing import Callable

from src.trajectory_piecewise_linear import TPWL  # type: ignore


# ============================================================
# Example systems
# ============================================================

class VanDerPol:
    """
    Van der Pol oscillator:
        dx1/dt = x2
        dx2/dt = mu*(1 - x1^2)*x2 - x1

    Nonlinear due to the (1-x1^2)*x2 term. Jacobian is analytic.
    """
    def __init__(self, mu: float = 1.5):
        self.mu = mu

    def f(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        mu = self.mu
        return np.array([x2,
                         mu * (1 - x1**2) * x2 - x1])

    def jac(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        mu = self.mu
        return np.array([
            [0,              1             ],
            [-2*mu*x1*x2 - 1, mu*(1 - x1**2)]
        ])


class Lorenz:
    """
    Lorenz system (chaotic):
        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def f(self, x: np.ndarray) -> np.ndarray:
        sx, sy, sz = x
        s, r, b = self.sigma, self.rho, self.beta
        return np.array([s*(sy - sx),
                         sx*(r - sz) - sy,
                         sx*sy - b*sz])

    def jac(self, x: np.ndarray) -> np.ndarray:
        sx, sy, sz = x
        s, r, b = self.sigma, self.rho, self.beta
        return np.array([
            [-s,        s,    0  ],
            [r - sz,   -1,  -sx  ],
            [sy,        sx,  -b  ],
        ])


class DuffingOscillator:
    """
    Duffing oscillator (nonlinear spring):
        dx1/dt = x2
        dx2/dt = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*cos(omega*t)

    Here we use the autonomous version (gamma=0) for a clean comparison.
    """
    def __init__(self, delta=0.3, alpha=-1.0, beta=1.0):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta

    def f(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        return np.array([
            x2,
            -self.delta*x2 - self.alpha*x1 - self.beta*x1**3
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        return np.array([
            [0,                              1         ],
            [-self.alpha - 3*self.beta*x1**2, -self.delta]
        ])


# ============================================================
# Plotting helpers
# ============================================================

def compare_trajectories(t, x_true, x_tpwl_near, x_tpwl_gauss,
                          expansion_points, system_name, state_labels):
    """Plot true vs TPWL trajectories and error."""
    n = x_true.shape[1]
    n_show = min(n, 2)
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, n_show, figure=fig, hspace=0.55, wspace=0.3)

    panel_bg = "#1a1d27"
    c_true   = "#58a6ff"
    c_near   = "#ff7b72"
    c_gauss  = "#3fb950"
    c_pts    = "#f0e442"

    def sa(ax, title, xl="", yl=""):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color="#e6edf3", fontsize=9.5, fontweight="bold", pad=5)
        ax.set_xlabel(xl, color="#8b949e", fontsize=8.5)
        ax.set_ylabel(yl, color="#8b949e", fontsize=8.5)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.grid(color="#21262d", alpha=0.8, lw=0.5)

    # Time-series panels
    for col in range(n_show):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, x_true[:, col], color=c_true, lw=1.8,
                label="Nonlinear (true)", zorder=4)
        ax.plot(t, x_tpwl_near[:, col], color=c_near, lw=1.2,
                ls="--", label="TPWL nearest", zorder=3)
        ax.plot(t, x_tpwl_gauss[:, col], color=c_gauss, lw=1.2,
                ls=":", label="TPWL Gaussian", zorder=3)
        sa(ax, f"{state_labels[col]} over time", "t",
           state_labels[col])
        if col == 0:
            ax.legend(fontsize=7.5, facecolor="#1a1d27",
                      labelcolor="#e6edf3", framealpha=0.9)

    # Error panels
    err_near  = np.linalg.norm(x_tpwl_near  - x_true, axis=1)
    err_gauss = np.linalg.norm(x_tpwl_gauss - x_true, axis=1)

    ax_err = fig.add_subplot(gs[1, :])
    ax_err.semilogy(t, err_near,  color=c_near,  lw=1.4,
                    label=f"||e||₂ TPWL nearest  (mean={err_near.mean():.4f})")
    ax_err.semilogy(t, err_gauss, color=c_gauss, lw=1.4,
                    label=f"||e||₂ TPWL Gaussian (mean={err_gauss.mean():.4f})")
    sa(ax_err, "State Error ||x_TPWL - x_true||₂  (log scale)", "t",
       "Error norm")
    ax_err.legend(fontsize=8, facecolor="#1a1d27", labelcolor="#e6edf3")

    # Phase portrait (2D or first two dims)
    ax_ph = fig.add_subplot(gs[2, :])
    ax_ph.plot(x_true[:, 0], x_true[:, 1], color=c_true, lw=1.6,
               label="True trajectory", zorder=4, alpha=0.9)
    ax_ph.plot(x_tpwl_near[:, 0],  x_tpwl_near[:, 1],  color=c_near,
               lw=1.1, ls="--", label="TPWL nearest", alpha=0.85)
    ax_ph.plot(x_tpwl_gauss[:, 0], x_tpwl_gauss[:, 1], color=c_gauss,
               lw=1.1, ls=":", label="TPWL Gaussian", alpha=0.85)

    pts = np.array(expansion_points)
    ax_ph.scatter(pts[:, 0], pts[:, 1], c=c_pts, s=35, zorder=6,
                  marker="x", linewidths=1.5, label=f"Expansion pts ({len(pts)})")
    sa(ax_ph, f"Phase Portrait — {system_name}  ({len(pts)} expansion points)",
       state_labels[0], state_labels[1])
    ax_ph.legend(fontsize=8, facecolor="#1a1d27", labelcolor="#e6edf3", ncol=2)

    fig.suptitle(
        f"TPWL: Trajectory Piecewise-Linear Approximation — {system_name}",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.01,
    )
    return fig


# ============================================================
# Main demonstration
# ============================================================

def run_demo(system, system_name, x0_train, x0_test, t_end,
            delta_train, state_labels, n_steps=800, output_path="pwla.png"):
    """
    Full TPWL workflow:
        1. Train on a representative trajectory from x0_train.
        2. Simulate TPWL (nearest + Gaussian) from x0_test.
        3. Simulate true nonlinear ODE from x0_test for comparison.
        4. Plot and report errors.
    """
    print(f"\n{'='*60}")
    print(f" {system_name}")
    print(f"{'='*60}")

    t_eval = np.linspace(0, t_end, n_steps)
    t_span = (0.0, t_end)
    ivp_kw = dict(method="RK45", rtol=1e-9, atol=1e-11)

    # ---- 1. True nonlinear solution (training trajectory) ----
    sol_train = solve_ivp(lambda t, x: system.f(x), t_span,
                          x0_train, t_eval=t_eval, **ivp_kw) # type: ignore
    assert sol_train.success, sol_train.message

    # ---- 2. Train TPWL (nearest) ----
    model_near = TPWL(system.f, system.jac, delta=delta_train,
            weighting="nearest")
    model_near.train(x0_train, t_span, t_eval, **ivp_kw)
    model_near.print_summary()

    # ---- 3. Train TPWL (Gaussian) — same expansion points ----
    model_gauss = TPWL(system.f, system.jac, delta=delta_train,
                       weighting="gaussian", alpha=3.0 / delta_train**2)
    model_gauss.train(x0_train, t_span, t_eval, **ivp_kw)

    # ---- 4. True nonlinear solution from test initial condition ----
    sol_true = solve_ivp(lambda t, x: system.f(x), t_span,
                         x0_test, t_eval=t_eval, **ivp_kw) # type: ignore
    assert sol_true.success, sol_true.message

    # ---- 5. TPWL simulations from test IC ----
    sol_near = model_near.simulate(x0_test, t_span, t_eval,
                                   method="RK45", rtol=1e-8, atol=1e-10)
    sol_gauss = model_gauss.simulate(x0_test, t_span, t_eval,
                                     method="RK45", rtol=1e-8, atol=1e-10)

    x_true  = sol_true.y.T
    x_near  = sol_near.y.T  if sol_near.success  else np.full_like(x_true, np.nan) # type: ignore
    x_gauss = sol_gauss.y.T if sol_gauss.success else np.full_like(x_true, np.nan) # type: ignore

    err_near  = np.linalg.norm(x_near  - x_true, axis=1)
    err_gauss = np.linalg.norm(x_gauss - x_true, axis=1)

    print(f"\n  Test trajectory errors (||x_TPWL - x_true||₂):")
    print(f"    Nearest  : mean={err_near.mean():.5f}, max={err_near.max():.5f}")
    print(f"    Gaussian : mean={err_gauss.mean():.5f}, max={err_gauss.max():.5f}")

    # ---- 6. Plot ----
    fig = compare_trajectories(
        t_eval, x_true, x_near, x_gauss,
        model_near.points, system_name, state_labels,
    )
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"\n  Figure saved: {output_path}")
    plt.close(fig)

    return {
        "model_near": model_near,
        "model_gauss": model_gauss,
        "x_true": x_true,
        "x_near": x_near,
        "x_gauss": x_gauss,
        "t": t_eval,
        "err_near": err_near,
        "err_gauss": err_gauss,
    }


# ============================================================
# Sensitivity to delta (expansion point density)
# ============================================================

def delta_sensitivity(system, system_name, x0_train, x0_test,
                      t_end, deltas, state_labels, output_path=None):
    """
    Show how accuracy changes with the expansion-point density parameter delta.
    Smaller delta → more points → better approximation (at higher cost).
    """
    print(f"\n  Delta sensitivity study for {system_name}...")
    t_eval = np.linspace(0, t_end, 600)
    t_span = (0.0, t_end)
    ivp_kw = dict(method="RK45", rtol=1e-9, atol=1e-11)

    sol_true = solve_ivp(lambda t, x: system.f(x), t_span, x0_test,
                         t_eval=t_eval, **ivp_kw)  # type: ignore
    x_true = sol_true.y.T

    results = []
    for delta in deltas:
        m = TPWL(system.f, system.jac, delta=delta, weighting="nearest")
        m.train(x0_train, t_span, t_eval, **ivp_kw)
        sol = m.simulate(x0_test, t_span, t_eval, method="RK45",
                         rtol=1e-8, atol=1e-10)
        x_approx = sol.y.T if sol.success else np.full_like(x_true, np.nan) # type: ignore
        err = np.linalg.norm(x_approx - x_true, axis=1).mean()
        results.append((delta, len(m.points), err))
        print(f"    delta={delta:.3f}  →  {len(m.points):3d} pts  "
              f"→  mean err={err:.5f}")

    deltas_arr, n_pts_arr, errs_arr = zip(*results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.grid(color="#21262d", alpha=0.8, lw=0.5)

    axes[0].loglog(deltas_arr, errs_arr, "o-", color="#58a6ff", lw=2, ms=7)
    axes[0].set_xlabel("delta (min point spacing)", color="#8b949e", fontsize=9)
    axes[0].set_ylabel("Mean ||error||₂", color="#8b949e", fontsize=9)
    axes[0].set_title("Accuracy vs delta\n(smaller delta → more points → lower error)",
                      color="#e6edf3", fontsize=9.5, fontweight="bold")

    axes[1].semilogx(n_pts_arr, errs_arr, "s-", color="#3fb950", lw=2, ms=7)
    axes[1].set_xlabel("Number of expansion points", color="#8b949e", fontsize=9)
    axes[1].set_ylabel("Mean ||error||₂", color="#8b949e", fontsize=9)
    axes[1].set_title("Accuracy vs number of expansion points",
                      color="#e6edf3", fontsize=9.5, fontweight="bold")

    fig.suptitle(f"TPWL: delta sensitivity — {system_name}",
                 color="#e6edf3", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"  Figure saved: {output_path}")
    plt.close(fig)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    # ---- Example 1: Van der Pol ----
    vdp = VanDerPol(mu=1.5)
    run_demo(
        system=vdp,
        system_name="Van der Pol (mu=1.5)",
        x0_train=np.array([2.0, 0.0]),
        x0_test=np.array([1.8, 0.5]),   # slightly different IC
        t_end=15.0,
        delta_train=0.4,
        state_labels=["x₁", "x₂"],
        n_steps=800,
        output_path="tpwl_vanderpol.png",
    )

    delta_sensitivity(
        system=vdp,
        system_name="Van der Pol",
        x0_train=np.array([2.0, 0.0]),
        x0_test=np.array([1.8, 0.5]),
        t_end=15.0,
        deltas=[2.0, 1.0, 0.6, 0.4, 0.25, 0.15, 0.08],
        state_labels=["x₁", "x₂"],
        output_path="tpwl_delta_sensitivity.png",
    )

    # ---- Example 2: Lorenz (chaotic) ----
    lorenz = Lorenz()
    run_demo(
        system=lorenz,
        system_name="Lorenz System (chaotic)",
        x0_train=np.array([1.0, 1.0, 1.0]),
        x0_test=np.array([1.0, 1.0, 1.0]),  # same IC — TPWL should track well
        t_end=8.0,
        delta_train=1.5,
        state_labels=["x", "y", "z"],
        n_steps=800,
        output_path="tpwl_lorenz.png",
    )

    # ---- Example 3: Duffing oscillator (lightly damped, double-well) ----
    # delta=0.05 (light damping) gives an oscillation that crosses both wells,
    # covering enough state space for TPWL to need several expansion points.
    duffing = DuffingOscillator(delta=0.05, alpha=-1.0, beta=1.0)
    run_demo(
        system=duffing,
        system_name="Duffing Oscillator (double-well, lightly damped)",
        x0_train=np.array([0.0, 1.5]),
        x0_test=np.array([0.1, 1.4]),
        t_end=30.0,
        delta_train=0.35,
        state_labels=["x₁ (displacement)", "x₂ (velocity)"],
        n_steps=800,
        output_path="tpwl_duffing.png",
    )

    print("\nAll done. Output files written to user-data/outputs/")