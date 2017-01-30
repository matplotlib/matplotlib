"""
Tests using patches with units.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

@image_comparison(baseline_images=['units_rectangle'], extensions=['png'])
def test_units_rectangle():
    import matplotlib.testing.jpl_units as U
    U.register()

    p = mpatches.Rectangle((5*U.km, 6*U.km), 1*U.km, 2*U.km)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.add_patch(p)
    ax.set_xlim([4*U.km, 7*U.km])
    ax.set_ylim([5*U.km, 9*U.km])

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
