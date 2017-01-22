from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import cleanup


@cleanup
def test_sankey():
    # lets just create a sankey instance and check the code runs
    sankey = Sankey()
    sankey.add()
