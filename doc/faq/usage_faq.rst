.. _usage-faq:

***************
Usage
***************

.. contents::
   :backlinks: none


.. _general_concepts:

General Concepts
================

:mod:`matplotlib` has an extensive codebase that can be daunting to many
new users. However, most of matplotlib can be understood with a fairly
simple conceptual framework and knowledge of a few important points.

All plotting is done by *axes*. In addition to the plots,
the axes objects are responsible for other components that handle axis
labeling, ticks, title, and plot legends. A *figure* is the container
for one or more axes objects.

Everything in matplotlib is organized in a heirarchy. At the top
of the heirarchy is the matplotlib state-machine environment. This
environment is responsible for managing the figures and axes
that have been created and modified by you. The behavior of the matplotlib
environment is similar to MATLAB and therefore should be most familiar to
users with MATLAB experience.

There are two interfaces to this environment: :mod:`pylab` and
:mod:`matplotlib.pyplot`. Through one of these two interfaces, the user
creates *figures*. These figures, in turn, create one or more *axes*.
These axes are then used for any plotting requested by you. Depending
on how you use matplotlib, these figures and axes can be created explicitly
by you through the interface::

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(range(10), range(10))
    ax.set_title("Simple Plot")
    plt.show()

or implicitly by the state-machine environment::

    import matplotlib.pyplot as plt

    plt.plot(range(10), range(10))
    plt.title("Simple Plot")
    plt.show()


.. _pylab:

Matplotlib, pylab, and pyplot: how are they related?
====================================================

Matplotlib is the whole package; :mod:`pylab` is a module in matplotlib
that gets installed alongside :mod:`matplotlib`; and :mod:`matplotlib.pyplot`
is a module in matplotlib.

Pyplot provides a state-machine interface to the underlying plotting
library in matplotlib.
For example, calling a plotting function from pyplot will
automatically create the necessary figure and axes to achieve
the desired plot. Setting a title through pyplot will automatically
set the title to the current axes object.

Pylab combines the pyplot functionality (for plotting) with the numpy
functionality (for mathematics and for working with arrays)
in a single namespace, making that namespace
(or environment) more MATLAB-like.
For example, one can call the `sin` and `cos` functions just like
you could in MATLAB, as well as having all the features of pyplot.

The pyplot interface is generally preferred for non-interactive plotting
(i.e., scripting). The pylab interface is convenient for interactive
calculations and plotting, as it minimizes typing. Note that this is
what you get if you use the *ipython* shell with the *-pylab* option,
which imports everything from pylab and makes plotting fully interactive.

.. _coding_styles:

Coding Styles
==================

When viewing this documentation and examples, you will find different
coding styles and usage patterns. These styles are perfectly valid
and have their pros and cons. Just about all of the examples can be
converted into another style and achieve the same results.
The only caveat is to avoid mixing the coding styles for your own code.

.. note::
   Developers for matplotlib have to follow a specific style and guidelines.
   See :ref:`developers-guide-index`.

Of the different styles, there are two that are officially supported.
Therefore, these are the preferred ways to use matplotlib.

For the preferred pyplot style, the imports at the top of your
scripts will typically be::

    import matplotlib.pyplot as plt
    import numpy as np

Then one calls, for example, np.arange, np.zeros, np.pi, plt.figure,
plt.plot, plt.show, etc. So, a simple example in this style would be::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

Note that this example used pyplot's state-machine to
automatically create a figure and an axes. For full control of
your plots and more advanced usage, use the pyplot interface
for creating figures, and then use the object methods for the rest::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()

Next, the same example using a pure MATLAB-style::

    from pylab import *
    x = arange(0, 10, 0.2)
    y = sin(x)
    plot(x, y)
    show()


So, why all the extra typing as one moves away from the pure
MATLAB-style?  For very simple things like this example, the only
advantage is academic: the wordier styles are more explicit, more
clear as to where things come from and what is going on.  For more
complicated applications, this explicitness and clarity become
increasingly valuable, and the richer and more complete object-oriented
interface will likely make the program easier to write and maintain.


