.. _usage-faq:

***************
Usage
***************

.. contents::
   :backlinks: none

.. _pylab:

Matplotlib, pylab, and pyplot: how are they related?
====================================================

Matplotlib is the whole package; :mod:`pylab` is a module in matplotlib
that gets
installed alongside :mod:`matplotlib`; and :mod:`matplotlib.pyplot` is a
module in matplotlib.

Pyplot provides a MATLAB-style state-machine interface to
the underlying object-oriented plotting library in matplotlib.

Pylab combines the pyplot functionality (for plotting) with the numpy
functionality (for mathematics and for working with arrays)
in a single namespace, making that namespace
(or environment) even more MATLAB-like.  This is what you get if
you use the
*ipython* shell with the *-pylab* option, which imports everything
from pylab and makes plotting fully interactive.

We have been gradually converting the matplotlib examples
from pure MATLAB-style, using "from pylab import \*", to a preferred
style in which pyplot is used for some convenience functions, either
pyplot or the object-oriented style is used for the remainder of the
plotting code, and numpy is used explicitly for numeric array operations.

In this preferred style, the imports at the top are::

    import matplotlib.pyplot as plt
    import numpy as np

Then one calls, for example, np.arange, np.zeros, np.pi, plt.figure,
plt.plot, plt.show, etc.

Example, pure MATLAB-style::

    from pylab import *
    x = arange(0, 10, 0.2)
    y = sin(x)
    plot(x, y)
    show()

Now in preferred style, but still using pyplot interface::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

And using pyplot convenience functions, but object-orientation for the rest::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()

So, why do all the extra typing required as one moves away from the pure
MATLAB-style?  For very simple things like this example, the only
advantage is educational: the wordier styles are more explicit, more
clear as to where things come from and what is going on.  For more
complicated applications, the explicitness and clarity become
increasingly valuable, and the richer and more complete object-oriented
interface will likely make the program easier to write and maintain.



