from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
import matplotlib.pyplot as plt

from matplotlib.testing.decorators import cleanup


@cleanup
def test_stem_remove():
    ax = plt.gca()
    st = ax.stem([1, 2], [1, 2])
    st.remove()
