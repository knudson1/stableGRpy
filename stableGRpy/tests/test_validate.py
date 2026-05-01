"""
Standalone tests for _validate.py.

Run the tests with:
    python -m pytest stablegrpy/tests/test_validate.py -v
or:
    python stablegrpy/tests/test_validate.py
"""

import numpy as np
import pytest

from stablegrpy._validate import validate_chains


class TestValidateChains:
    def test_single_1d_array(self):
        out = validate_chains(np.random.randn(100))
        assert len(out) == 1
        assert out[0].shape == (100, 1)

    def test_single_2d_array(self):
        out = validate_chains(np.random.randn(100, 3))
        assert len(out) == 1
        assert out[0].shape == (100, 3)

    def test_list_of_arrays(self):
        chains = [np.random.randn(100, 2) for _ in range(4)]
        out = validate_chains(chains)
        assert len(out) == 4
        assert all(c.shape == (100, 2) for c in out)

    def test_autoburnin_halves_length(self):
        x = np.arange(200).reshape(100, 2).astype(float)
        out = validate_chains(x, autoburnin=True)
        assert out[0].shape[0] == 50

    def test_autoburnin_keeps_second_half(self):
        x = np.arange(200).reshape(100, 2).astype(float)
        out = validate_chains(x, autoburnin=True)
        assert out[0][0, 0] == 100.0    # first row of second half

    def test_output_dtype_is_float64(self):
        x = np.ones((50, 2), dtype=np.int32)
        out = validate_chains(x)
        assert out[0].dtype == np.float64

    def test_tuple_of_arrays_accepted(self):
        chains = tuple(np.random.randn(100, 2) for _ in range(3))
        out = validate_chains(chains)
        assert len(out) == 3

    def test_mismatched_rows_raises(self):
        chains = [np.random.randn(100, 2), np.random.randn(80, 2)]
        with pytest.raises(ValueError, match="same number of iterations"):
            validate_chains(chains)

    def test_mismatched_cols_raises(self):
        chains = [np.random.randn(100, 2), np.random.randn(100, 3)]
        with pytest.raises(ValueError, match="same number of variables"):
            validate_chains(chains)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            validate_chains(np.array([1.0, 2.0, 3.0]))

    def test_unrecognised_type_raises(self):
        with pytest.raises((TypeError, Exception)):
            validate_chains("not an array")

    def test_single_chain_single_variable(self):
        out = validate_chains(np.random.randn(50, 1))
        assert out[0].shape == (50, 1)

    def test_many_chains(self):
        chains = [np.random.randn(200, 4) for _ in range(10)]
        out = validate_chains(chains)
        assert len(out) == 10
        assert all(c.shape == (200, 4) for c in out)


if __name__ == "__main__":
    # Run without pytest: python tests/test_validate.py
    import sys
    passed = failed = 0
    suite = TestValidateChains()
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