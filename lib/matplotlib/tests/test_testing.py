import warnings
import pytest

@pytest.mark.xfail(strict=True,
                   reason="testing that warnings fail tests")
def test_warn_to_fail():
    warnings.warn("This should fail the test")
