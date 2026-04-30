"""
Tests for mvn_gibbs.py.

Run the tests with:
    python -m pytest stablegrpy/tests/test_mvn_gibbs.py -v
or:
    python stablegrpy/tests/test_mvn_gibbs.py

"""

import numpy as np
import pytest

from stablegrpy.mvn_gibbs import mvn_gibbs

MU = np.array([1.0, 1.0])
SIGMA = np.array([[1.0, 0.8], [0.8, 1.0]])

class TestMvnGibbs:
    def test_output_shape(self):
        chain = mvn_gibbs(n=500, mu=MU, sigma=SIGMA,
                          rng=np.random.default_rng(0))
        assert chain.shape == (500, 2)

    def test_mean_close_to_truth(self):
        rng = np.random.default_rng(7)
        chain = mvn_gibbs(n=50000, mu=MU, sigma=SIGMA, rng=rng)
        assert np.allclose(chain.mean(axis=0), MU, atol=0.05)

    def test_cov_close_to_truth(self):
        rng = np.random.default_rng(7)
        chain = mvn_gibbs(n=50000, mu=MU, sigma=SIGMA, rng=rng)
        assert np.allclose(np.cov(chain, rowvar=False), SIGMA, atol=0.05)

    def test_min_dimension(self):
        with pytest.raises(ValueError):
            mvn_gibbs(n=100, mu=np.array([0.0]), sigma = np.eye(1))

if __name__ == "__main__":
    # Run without pytest: python stablegrpy/tests/test_mvn_gibbs.py
    import sys
    passed = failed = 0
    suite = TestMvnGibbs()
    tests = [m for m in dir(suite) if m.startswith("test_")]
    for name in tests:
        try:
            getattr(suite, name)()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(0 if failed == 0 else 1)