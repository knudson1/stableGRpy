"""
Two-block Gibbs sampler for a multivariate normal distribution.

Used primarily for examples and tests, matching the ``mvn.gibbs`` function
in the R package stableGR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mvn_gibbs(
    n: int,
    mu: NDArray,
    sigma: NDArray,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """
    Sample from a multivariate normal distribution using a two-block Gibbs
    sampler.

    The chain is split into two blocks: the first floor(p/2) variables and
    the remaining variables. Each is updated by sampling from its full
    conditional distribution given the other block.

    This function is provided primarily for use in examples and tests,
    mirroring the ``mvn.gibbs`` function in the R package stableGR.

    Parameters
    ----------
    n : int
        Number of MCMC iterations to generate.
    mu : array-like, shape (p,)
        Mean vector of the target multivariate normal distribution.
    sigma : array-like, shape (p, p)
        Covariance matrix of the target distribution.
    rng : numpy.random.Generator or None, optional
        Random number generator. If None, uses ``np.random.default_rng()``.

    Returns
    -------
    ndarray, shape (n, p)
        Array of MCMC samples. Each row is one iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from stablegrpy import mvn_gibbs
    >>> sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    >>> chain = mvn_gibbs(n=1000, mu=np.ones(2), sigma=sigma)
    >>> chain.shape
    (1000, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    p = len(mu)

    if p < 2:
        raise ValueError(
            f"A block Gibbs sampler will not work with less than 2 parameters, but you provided {p}"
        )

    if sigma.shape != (p, p):
        raise ValueError(
            f"sigma must be shape ({p}, {p}), but you provided {sigma.shape}"
        )

    # Create indices for the two blocks
    p1 = p // 2
    idx1 = np.arange(0, p1)        # first block has first floor(p/2) parameters
    idx2 = np.arange(p1, p)        # second block has remaining parameters

    mu1, mu2 = mu[idx1], mu[idx2]

    # Split the covariance matrix up into 4 pieces 
    sig11 = sigma[np.ix_(idx1, idx1)] # cov matrix of first block, dim p1 x p1
    sig12 = sigma[np.ix_(idx1, idx2)] # var matrix of first block with second, dim p1 x p2
    sig21 = sigma[np.ix_(idx2, idx1)] # var matrix of second block with first, dim p2 x p1
    sig22 = sigma[np.ix_(idx2, idx2)] # cov matrix of second block, dim p2 x p2

    # Precompute conditional covs, to be used in loop later for efficiency
    # For first block, conditional on second
    # X1 | X2 ~ N(mu1 + sig12 sig22^{-1} (X2 - mu2),  sig11 - sig12 sig22^{-1} sig21)
    sig22_inv = np.linalg.inv(sig22)
    A = sig12 @ sig22_inv               # (p1, p2)
    cond_cov_1 = sig11 - A @ sig21      # (p1, p1)

    # For second block, conditional on first
    # X2 | X1 ~ N(mu2 + sig21 sig11^{-1} (X1 - mu1),  sig22 - sig21 sig11^{-1} sig12)
    sig11_inv = np.linalg.inv(sig11)
    B = sig21 @ sig11_inv               # (p2, p1)
    cond_cov_2 = sig22 - B @ sig12      # (p2, p2)

    # Cholesky decomposition (LL^T) for efficient sampling (this is built into R's rmvnorm)
    L1 = np.linalg.cholesky(cond_cov_1)
    L2 = np.linalg.cholesky(cond_cov_2)

    samples = np.zeros((n, p))
    x = np.zeros(p)     # Start the chain at zero 

    # Generate the markov chain
    for t in range(n):
        # update block 1 | block 2
        cond_mean_1 = mu1 + A @ (x[idx2] - mu2)
        x[idx1] = cond_mean_1 + L1 @ rng.standard_normal(len(idx1))

        # update block 2 | block 1
        cond_mean_2 = mu2 + B @ (x[idx1] - mu1)
        x[idx2] = cond_mean_2 + L2 @ rng.standard_normal(len(idx2))

        samples[t] = x

    return samples
