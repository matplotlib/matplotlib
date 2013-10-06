from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup


@cleanup
def test_savefig_to_stringio():
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(six.moves.StringIO(), format="ps")


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
