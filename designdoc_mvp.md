**stablegrpy Design Document**

- [1. Overview](#1-overview)
- [2. MVP Scope](#2-mvp-scope)
- [3. File Structure](#3-file-structure)
- [4. Functions and Classes](#4-functions-and-classes)
  - [4.1. `mvn_gibbs.py`](#41-mvn_gibbspy)
  - [4.2. `_validate.py`](#42-_validatepy)
  - [4.3. `_variance.py`](#43-_variancepy)
  - [4.4. `stable_gr.py`](#44-stable_grpy)
- [5. Call Graph](#5-call-graph)
- [6. Development Order](#6-development-order)
- [7. Future Arguments](#7-future-arguments)
- [8. Notable Differences from the R Package](#8-notable-differences-from-the-r-package)
- [9. R and Python Package Concepts](#9-r-and-python-package-concepts)


# 1. Overview

This is the start of the Python port of the R package `stableGR`. This implements the improved Gelman-Rubin convergence diagnostic using the replicated lugsail batch means variance estimator. 

The user will provide the MCMC sample (as numpy array(s)) and this will return the multivariate PSRF (aka the Gelman-Rubin convergence diagnostic calculated with the lugsail batch means variance estimate).

---

# 2. MVP Scope

The minimum viable product mirrors the defaults of `stable.GR()` in the R package. One public function, one estimator, hardcoded defaults set as internal variables so they can be promoted to arguments later without restructuring the code.

Depends only on `numpy`.

Out of scope for MVP:
- `n_eff()` and `target_psrf()` (requires `scipy`)
- `asym_var()` as a standalone public function
- Alternative variance estimators (`bm`, `obm`)
- Alternative mappings (besides the determinant)
- Custom batch sizes exposed as arguments

---

# 3. File Structure

```
stablegrpy/
├── _validate.py        input validation
├── _variance.py        lugsail variance estimator
├── mvn_gibbs.py        test/demo MCMC sampler
├── stable_gr.py        Gelman-Rubin diagnostic (sole public function)
└── __init__.py         public API
```

---

# 4. Functions and Classes

The MVP has just one class, `GRResult`, which is a dataclass. It serves as the return type of `stable_gr()`, bundling the diagnostic outputs into a single named object rather than returning a plain tuple or dictionary. In future versions, `NEFFResult` will be added as the return type of `n_eff()`. All other components of the package are functions.

---

## 4.1. `mvn_gibbs.py`

**`mvn_gibbs(n, mu, sigma, rng=None)`**

Two-block Gibbs sampler for a multivariate normal. Used for examples and tests. Requires `p >= 2`.

| | |
|---|---|
| **Input** | `n` — int, number of iterations |
| | `mu` — `ndarray (p,)`, target mean |
| | `sigma` — `ndarray (p, p)`, target covariance |
| | `rng` — `numpy.random.Generator` or None |
| **Output** | `ndarray (n, p)` |
| **Raises** | `ValueError` if `p < 2` |
| **Called by** | tests and user examples only |

---


## 4.2. `_validate.py`

**`validate_chains(chains, autoburnin=False)`**

Normalizes raw user input into a consistent internal format.

| | |
|---|---|
| **Input** | `chains` — a 1-D array, 2-D array, or list of 2-D arrays |
| | `autoburnin` — bool; if True, discards first half of each chain |
| **Output** | list of `ndarray`, each shape `(n, p)`, dtype `float64` |
| **Raises** | `ValueError` if chains have mismatched shapes or fewer than 4 iterations |
| **Called by** | `stable_gr` |

---

## 4.3. `_variance.py`

Internal module. Implements the lugsail batch means estimator.

---

**`_parse_batch_size(n, size)`**

Resolves the batch size to a concrete integer.

| | |
|---|---|
| **Input** | `n` — int, number of iterations |
| | `size` — None, `'sqroot'`, `'cuberoot'`, or int |
| **Output** | int |
| **Called by** | `lugsail_variance` |

---

**`_trim_chains(chains, b)`**

Trims each chain so its length is an exact multiple of `b`.

| | |
|---|---|
| **Input** | `chains` — list of `ndarray (n, p)` |
| | `b` — int, batch size |
| **Output** | trimmed list of `ndarray`, n_trim, a (number of batches) |
| **Called by** | `lugsail_variance` |

---

**`_replicated_bm(chains, b, overall_mean)`**

Computes the replicated batch means matrix T̂_b.

| | |
|---|---|
| **Input** | `chains` — list of `ndarray (n, p)`, already trimmed to multiple of `b` |
| | `b` — int, batch size |
| | `overall_mean` — `ndarray (p,)` |
| **Output** | `ndarray (p, p)` |
| **Called by** | `lugsail_variance` |

---

**`lugsail_variance(chains, size, multivariate)`**

Replicated lugsail batch means estimator: T̂_L = 2T̂_b − T̂_{b//3}

| | |
|---|---|
| **Input** | `chains` — list of `ndarray (n, p)` |
| | `size` — None, `'sqroot'`, `'cuberoot'`, or int |
| | `multivariate` — bool; if False returns diagonal only |
| **Output** | `ndarray (p, p)` if `multivariate=True`, else `ndarray (p,)` |
| **Calls** | `_parse_batch_size`, `_trim_chains`, `_replicated_bm` |
| **Called by** | `stable_gr` |

---

## 4.4. `stable_gr.py`

**`GRResult`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `psrf` | `ndarray (p,)` | univariate PSRF per variable |
| `psrf_multivariate` | float or None | multivariate PSRF; None if `p=1` |
| `means` | `ndarray (p,)` | grand mean across all chains |

---

**`stable_gr(chains, autoburnin=False)`**

Computes the stable Gelman-Rubin diagnostic. Internal variables are set at the top of the function rather than exposed as arguments, so they can be promoted to arguments later without restructuring.

Internal variables (set at top of function):
```python
method = "lug"
mapping = "determinant"
size = None
multivariate = True
```

| | |
|---|---|
| **Input** | `chains` — a 1-D array, 2-D array, or list of 2-D arrays |
| | `autoburnin` — bool; if True, discards first half of each chain |
| **Output** | `GRResult` |
| **Calls** | `validate_chains`, `lugsail_variance` |

---

# 5. Call Graph

```
stable_gr ──┬── validate_chains
            └── lugsail_variance ──┬── _parse_batch_size
                                   ├── _trim_chains
                                   └── _replicated_bm
```

---

# 6. Development Order

**Step 1 — `_validate.py`**
Write first. Defines the input contract that everything else depends on. Test with `tests/test_validate.py`. Done.

**Step 2 — `mvn_gibbs.py`**
Write second. Needed to generate ground-truth MCMC samples for testing all downstream functions. Test with `tests/test_mvn_gibbs.py`. Done.


**Step 3 — `_variance.py`**
Implement `_replicated_bm` and `lugsail_variance`. Verify on iid samples where the asymptotic variance is known analytically. Test with `tests/test_variance.py`.

**Step 4 — `stable_gr.py`**
Depends on `_validate.py` and `_variance.py`. Verify PSRF approaches 1 for long well-mixed chains. Test with `tests/test_stable_gr.py`.

**Step 5 — `__init__.py`**
Wire up the public API. Export `stable_gr` and `GRResult`.

**Step 6 — README.md and packaging**
Write after the API has stabilised. Verify `pip install -e .` works cleanly and all tests pass before publishing.

---

# 7. Future Arguments

When ready to expose the internal variables as arguments, the function signature changes from:

```python
def stable_gr(chains, autoburnin=False):
```

to:

```python
def stable_gr(chains, autoburnin=False, method="lug", mapping="determinant",
              size=None, multivariate=True):
```

No other code changes are needed since the rest of the function already references these as variables.

---

# 8. Notable Differences from the R Package

| Difference | R | Python MVP |
|---|---|---|
| `mvn.gibbs` starting value | zero | not included in MVP |
| `mvn.gibbs` update order | block 2 then block 1 | not included in MVP |
| `p = 1` in `mvn.gibbs` | works (splits into p-1 and 1) | not included in MVP |
| Return type | named list | `GRResult` dataclass |
| `n.eff` and `target.psrf` | included | out of scope for MVP |
| Alternative methods | `bm`, `obm`, `tukey`, `bartlett` | out of scope for MVP |
| Dependencies | `mcmcse` | none beyond `numpy` (`mcmcse` does not have a python equivalent) |

# 9. R and Python Package Concepts

| R Concept | Python Equivalent | Notes |
|---|---|---|
| `NAMESPACE` file | `__init__.py` | Both control what is exported/public. R uses explicit `export()` calls in `NAMESPACE`; Python uses imports in `__init__.py`. Neither strictly prevents users from accessing internals. |
| Unexported function | Function with leading `_` | In R, unexported functions are accessed via `:::`. In Python, `_private` functions can still be imported directly — the underscore is convention only. |
| `DESCRIPTION` file | `pyproject.toml` | Both declare package name, version, authors, license, and dependencies. `DESCRIPTION` uses `Imports:` for dependencies; `pyproject.toml` uses `dependencies`. |
| `Imports:` in DESCRIPTION | `dependencies` in `pyproject.toml` | Packages that are automatically installed when a user installs your package. |
| `Suggests:` in DESCRIPTION | `[project.optional-dependencies]` in `pyproject.toml` | Packages that are useful but not required — e.g. testing or documentation tools. In Python installed via `pip install stablegrpy[dev]`. |
| `.R` source file | `.py` source file | Direct equivalent. One module per file is idiomatic in both languages. |
| `R/` directory | Package directory (e.g. `stablegrpy/`) | Both are where source files live. R requires the folder be named `R/`; Python requires the folder match the package name and contain `__init__.py`. |
| `tests/testthat/` | `tests/` | Both are where tests live. R uses `testthat`; Python typically uses `pytest`. |
| `testthat::test_that()` | `pytest` test function or class | R groups tests with `test_that("description", { ... })`; Python uses functions prefixed with `test_` or classes prefixed with `Test`. |
| `expect_equal()` | `assert` statement | R's `expect_*` functions check conditions inside tests; Python uses plain `assert` or `pytest`'s assertion introspection. |
| `devtools::load_all()` | `pip install -e .` | Both reload the package from source during development without a full reinstall. Changes to source files are reflected immediately. |
| `devtools::install()` | `pip install .` | Installs the package from local source. Unlike the editable versions above, changes to source are not reflected until you reinstall. |
| CRAN | PyPI | The central public package repository. `install.packages()` in R; `pip install` in Python. |
| `devtools::check()` | `pytest` + `twine check` | R's check runs tests, examples, and structural validation before CRAN submission. Python separates these: `pytest` for tests, `twine check dist/*` for package validity before PyPI upload. |
| `roxygen2` docstring | NumPy/Google style docstring | Both are conventions for writing inline documentation that can be rendered into formatted docs. R uses `#'` comments above functions; Python uses triple-quoted strings inside functions. |
| `pkgdown` | `sphinx` | Both generate a documentation website from source files and docstrings. |
| `.Rproj` file | `pyproject.toml` + `.venv/` | R projects are organised around an `.Rproj` file; Python projects around `pyproject.toml` with a virtual environment managing dependencies. Not a perfect equivalence but serves a similar organisational role. |
| `library()` / `require()` | `import` | Both load a package into the current session. `library()` errors on failure; `require()` returns FALSE. Python's `import` errors on failure; `importlib` can be used for conditional imports. |
| `:::` operator | Direct `_module` import | Both access internals that aren't part of the public API. `pkg:::internal_fn()` in R; `from pkg._module import fn` in Python. Both work but are discouraged for user code. |
| `environment()` / namespaces | Module scope | R uses environments to manage where functions look up variables. Python uses module-level scope. Both determine what names are visible inside a function. |
