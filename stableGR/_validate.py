"""
Input validation for MCMC chain data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_chains(
    chains,
    autoburnin: bool = False,
) -> list[NDArray]:
    """
    Validate and normalize MCMC chain input.

    Accepts:
    - A single 1-D array (one chain, one variable)
    - A single 2-D array (one chain, multiple variables)
    - A list of 1-D or 2-D arrays (multiple chains)

    All chains must have the same number of iterations (rows) and the same
    number of variables (columns).

    Parameters
    ----------
    chains : array-like or list of array-like
        The MCMC samples.
    autoburnin : bool, optional
        If True, discard the first half of each chain (use only the second
        half), matching the ``autoburnin`` option in the R package.

    Returns
    -------
    list of ndarray, shape (n, p)
        Validated list of chain arrays, each with dtype float64.

    Raises
    ------
    ValueError
        If inputs are inconsistent or invalid.
    TypeError
        If the input type is not recognised.
    """
    # ---- normalize to list of 2-D arrays --------------------------------
    if isinstance(chains, np.ndarray):
        chains = [chains]
    elif not isinstance(chains, (list, tuple)):
        # Try converting (e.g. pandas DataFrame, torch tensor …)
        try:
            chains = [np.asarray(chains, dtype=float)]
        except Exception:
            raise TypeError(
                "chains must be a numpy array, or a list/tuple of numpy arrays."
            )

    out = []
    for i, c in enumerate(chains):
        arr = np.asarray(c, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(
                f"Chain {i} has {arr.ndim} dimensions; expected 1 or 2."
            )
        out.append(arr)

    if len(out) == 0:
        raise ValueError("chains must contain at least one chain.")

    n_iter = out[0].shape[0]
    n_var = out[0].shape[1]

    for i, arr in enumerate(out[1:], start=1):
        if arr.shape[0] != n_iter:
            raise ValueError(
                f"All chains must have the same number of iterations. "
                f"Chain 0 has {n_iter} rows, chain {i} has {arr.shape[0]} rows."
            )
        if arr.shape[1] != n_var:
            raise ValueError(
                f"All chains must have the same number of variables. "
                f"Chain 0 has {n_var} columns, chain {i} has {arr.shape[1]} columns."
            )

    if n_iter < 4:
        raise ValueError(
            f"Each chain must have at least 4 iterations; got {n_iter}."
        )

    # ---- optional auto burn-in ------------------------------------------
    if autoburnin:
        start = n_iter // 2
        out = [arr[start:] for arr in out]

    return out
