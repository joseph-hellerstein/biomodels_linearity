# Evaluation Metrics

## Proposed metric

This is the per-point MAPE-based forecast accuracy with a floor at zero. MAPE is minus the absolute percetage error.

$max(0, 1 - \frac{| y - \hat{y} |}{y})$

## Claude's Assessment

This is a legitimate loss function, specifically $L(\hat y) = \min!\left(1, |y-\hat y|/y\right)$: absolute error weighted by $1/y$ and then Winsorized at 1. That's a coherent object with a real literature behind it.

What persists, in mirrored form. Under-prediction now gives $S = \rho$, which decays smoothly and only reaches zero at $\hat y = 0$. Over-prediction gives $S = 2 - \rho$, which hits zero at $\hat y = 2y$ and stays there. So the direction of the bias has reversed: the metric now favors forecasting low.Concretely: a forecast 10× too low scores 0.1, while a forecast 3× too high scores 0. Systematic hedging downward is rewarded.

This metric has a name, and a known bias $1 - |y-\hat y|/y$ is one minus the absolute percentage error, so your metric is per-point MAPE-based "forecast accuracy" with a floor at zero — the convention used routinely in demand planning and supply chain ("we hit 82% forecast accuracy"). Its properties are well studied and mostly cautionary.

The key result is Gneiting's (2011, JASA, "Making and Evaluating Point Forecasts"): minimizing expected APE does not elicit the mean or the median. It is consistent for the $\beta$-median with $\beta = -1$ — the median of the distribution whose density is proportional to $y^{-1}f(y)$. Because that reweighting tilts mass toward small values, the optimal forecast under this metric sits below the median, and well below the mean for right-skewed data. Since most positive-valued time series (demand, traffic, counts, prices, energy) are right-skewed, the bias is not a corner case.

Your clipping strengthens this. Take $y \in {1, 10}$ with equal probability. Predicting the conditional mean, 5.5, scores 0.275. Predicting the constant 1 scores 0.55; predicting the constant 10 scores 0.50. The mean forecast is beaten by both constants — a calibrated forecaster loses to a biased one, and the metric's preferred answer is the low mode. If you're ranking predictors, one that is systematically conservative can win despite being worse in any decision-theoretic sense you'd articulate independently.

Where the clipping helps and where it hurts
Worth separating, because it does both.

Helps: it bounds influence. Raw MAPE has no upper limit per observation, so a single blowup at a small $y$ can dominate an average over thousands of points and make a mean MAPE meaningless. Clipping at 1 turns your aggregate into a censored mean, which is far more robust — structurally the same idea as a redescending M-estimator (Tukey biweight, Hampel). For ranking predictors on messy data this is a real advantage over reporting mean MAPE, and note that averaging $\max(0, 1-\text{APE})$ pointwise is not the same as $1 - \overline{\text{APE}}$; yours is the better-behaved of the two.

Hurts: it destroys resolution above $2y$ and makes the loss non-convex with zero gradient there. Every over-prediction beyond 2× is scored identically, so you cannot distinguish a predictor that's occasionally 2.5× high from one that's occasionally 50× high. If a meaningful fraction of your points land in the flat region — check this — the metric is comparing predictors on a censored subsample without telling you. Report the saturation rate alongside the mean score; if it's more than a few percent, the ranking is fragile.

The $1/y$ weighting is the subtler issue for your use case
Scale invariance is what you wanted, and you have it: multiply $y$ and $\hat y$ by any constant and $S$ is unchanged, so scores are commensurable across variables of vastly different magnitude. But relative error normalizes by the level, not the variability, which has two consequences worth checking:

Within a series with wide dynamic range, low-level time points get weight $1/y$ and dominate the average. A series that dips to a tenth of its typical value contributes ten times the sensitivity at those moments, so your score is largely determined by performance in troughs. If the troughs are the hard-to-predict part, every predictor looks similar; if they're the easy part, everything looks good.

Across series, a variable that is intrinsically volatile relative to its mean will score poorly no matter how good the predictor is, while a near-constant variable scores near 1 for a naïve forecast. Averaging across variables then ranks predictors partly by which variables happen to be easy.

Alternatives, and what I'd actually suggest
MASE (Hyndman & Koehler 2006) scales absolute error by the in-sample MAE of a naïve or seasonal-naïve forecast on that same series. This is the standard fix for cross-series comparability and it addresses the weighting problem directly: the normalizer is the series' own scale of variation, so troughs don't get inflated weight and intrinsically volatile series aren't penalized for being volatile. It's also symmetric and consistent for the median. Its main cost is interpretability — 0.8 means "20% better than naïve," not "80% accurate."
Log accuracy ratio, $\log(\hat y/y)$, summarized by median and median absolute deviation (Morley et al. 2018). Symmetric in ratio, scale-invariant, and cleanly separates bias (median) from spread (MAD). Since $1 - |\rho - 1| \approx 1 - |\log \rho|$ near $\rho = 1$, your metric is a linearization of this that inherits its asymmetry from the approximation.
sMAPE halves the asymmetry by using $(y+\hat y)/2$ as denominator, but it's not genuinely symmetric and I'd skip it in favor of the above.
CRPS, or the energy / variogram scores, if your predictors produce distributions. These are proper, so they can't be gamed by hedging. The variogram score is specifically worth knowing for multivariate work because it rewards correct cross-variable dependence — something a per-point metric ignores entirely, along with all temporal structure. If "quality of predictor" includes getting the joint behavior right, no pointwise score will capture it.
If you want to keep a bounded $[0,1]$ interpretable score, the minimal repair is still

$$S = \frac{\min(y, \hat y)}{\max(y, \hat y)} = e^{-|\log(\hat y/y)|}$$

Scale-invariant, bounded, equals 1 exactly at $\hat y = y$, symmetric between over- and under-prediction, and never saturates — so 10× off in either direction scores 0.1, and you retain resolution among the bad cases. It's also bounded in influence, keeping the robustness benefit of your clipping without the flat region.

Whichever you choose, I'd report a bias diagnostic separately — median $\log(\hat y / y)$ per variable is cheap and immediately shows whether a predictor is winning by hedging low. A single accuracy number that's biased in a known direction will quietly select for predictors exploiting that direction, and with your current metric that direction is downward.