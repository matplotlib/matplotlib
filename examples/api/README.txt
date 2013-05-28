matplotlib API
==============

These examples use the matplotlib api rather than the pylab/pyplot
procedural state machine.  For robust, production level scripts, or
for applications or web application servers, we recommend you use the
matplotlib API directly as it gives you the maximum control over your
figures, axes and plottng commands.

The example agg_oo.py is the simplest example of using the Agg backend
which is readily ported to other output formats.  This example is a
good starting point if your are a web application developer.  Many of
the other examples in this directory use matplotlib.pyplot just to
create the figure and show calls, and use the API for everything else.
This is a good solution for production quality scripts.  For full
fledged GUI applications, see the user_interfaces examples.

Example style guide
===================

If you are creating new examples, you cannot import pylab or import *
from any module in your examples.  The only three functions allowed
from pyplot are "figure", "show" and "close", which you can use as
convenience functions for managing figures.  All other matplotlib
functionality must illustrate the API.

A simple example of the recommended style is::

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))
    ax.set_xlabel('some x data')
    ax.set_ylabel('some y data')
    ax.set_title('some title')
    ax.grid(True)
    fig.savefig('myfig')
    plt.show()
