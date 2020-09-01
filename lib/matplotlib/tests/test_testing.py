import warnings
import pytest
from matplotlib.testing.decorators import check_figures_equal


@pytest.mark.xfail(
    strict=True, reason="testing that warnings fail tests"
)
def test_warn_to_fail():
    warnings.warn("This should fail the test")


@pytest.mark.parametrize("a", [1])
@check_figures_equal(extensions=["png"])
@pytest.mark.parametrize("b", [1])
def test_parametrize_with_check_figure_equal(a, fig_ref, b, fig_test):
    assert a == b


def test_wrap_failure():
    with pytest.raises(ValueError, match="^The decorated function"):
        @check_figures_equal()
        def should_fail(test, ref):
            pass
