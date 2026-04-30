# stablegrpy

A stable Gelman-Rubin convergence diagnostic for Markov chain Monte Carlo.

Python port of the R package [stableGR](https://cran.r-project.org/package=stableGR) (Knudson & Vats), implementing the improved Gelman-Rubin statistic based on the **replicated lugsail batch means estimator**.

## Background

The classical Gelman-Rubin R̂ statistic uses the between-chain variance B to estimate the Monte Carlo variance — an estimator whose efficiency *decreases* without bound relative to better alternatives as chain length grows. This package replaces B with the **lugsail batch means estimator** (Vats & Flegal, 2022), which:

- Is biased from above in finite samples, protecting against premature convergence declarations
- Is strongly consistent for the true asymptotic variance
- Has relative efficiency over B that grows without bound as n → ∞
- Works for a single chain (m = 1)

Additionally, this package provides a **principled, ESS-based stopping threshold** that replaces the arbitrary R̂ ≤ 1.1 cutoff. The threshold is derived from the one-to-one relationship between PSRF and effective sample size (Vats & Knudson, 2021):

```
R̂ ≈ sqrt(1 + m / ESS)
```

A threshold of R̂ ≤ 1.1 corresponds to only ~5 effective samples per chain — far too few for reliable inference.

## Installation

```bash
pip install "git+https://github.com/knudson1/stableGR.git#subdirectory=stableGRpy"```

**Dependencies:** `numpy >= 1.23`, `scipy >= 1.9`

## References

Vats, D. and Knudson, C. (2021). Revisiting the Gelman-Rubin Diagnostic. *Statistical Science*, 36(4), 518–529. [arXiv:1812.09384](https://arxiv.org/abs/1812.09384)

Vats, D. and Flegal, J. M. (2022). Lugsail lag windows for estimating time-average covariance matrices. *Biometrika*, 109(3), 735–750. [arXiv:1809.04541](https://arxiv.org/abs/1809.04541)

Vats, D., Flegal, J. M., and Jones, G. L. (2019). Multivariate output analysis for Markov chain Monte Carlo. *Biometrika*, 106(2), 321–337.

Gelman, A. and Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457–472.

Brooks, S. P. and Gelman, A. (1998). General methods for monitoring convergence of iterative simulations. *Journal of Computational and Graphical Statistics*, 7(4), 434–455.

## License

GPL-3.0-or-later (matches the license of the original R package stableGR).
