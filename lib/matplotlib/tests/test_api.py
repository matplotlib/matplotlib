import re

import numpy as np
import pytest

from matplotlib import _api


@pytest.mark.parametrize('target,test_shape',
                         [((None, ), (1, 3)),
                          ((None, 3), (1,)),
                          ((None, 3), (1, 2)),
                          ((1, 5), (1, 9)),
                          ((None, 2, None), (1, 3, 1))
                          ])
def test_check_shape(target, test_shape):
    error_pattern = (f"^'aardvark' must be {len(target)}D.*" +
                     re.escape(f'has shape {test_shape}'))
    data = np.zeros(test_shape)
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)


def test_classproperty_deprecation():
    class A:
        @_api.deprecated("0.0.0")
        @_api.classproperty
        def f(cls):
            pass
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        A.f
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        a = A()
        a.f
