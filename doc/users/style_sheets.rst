.. _style-sheets:

***********************************
Customizing plots with style sheets
***********************************


The ``style`` package adds support for easy-to-switch plotting "styles" with
the same parameters as a matplotlibrc_ file.

There are a number of pre-defined styles provided by matplotlib. For
example, there's a pre-defined style called "ggplot", which emulates the
aesthetics of ggplot_ (a popular plotting package for R_). To use this style,
just add::

   >>> import matplotlib.pyplot as plt
   >>> plt.style.use('ggplot')

To list all available styles, use::

   >>> print plt.style.available


Defining your own style
=======================

You can create custom styles and use them by calling ``style.use`` with the
path or URL to the style sheet. Alternatively, if you add your ``<style-name>.mplstyle`` 
file to ``mpl_configdir/stylelib`, you can reuse your custom style sheet with a call to 
``style.use(<style-name>)``. By default ``mpl_configdir`` should be ``~/.config/matplotlib``, 
but you can check where yours is with ``matplotlib.get_configdir()``, you may need to 
create this directory. Note that a custom style sheet in ``mpl_configdir/stylelib`` 
will override a style sheet defined by matplotlib if the styles have the same name.

For example, you might want to create
``mpl_configdir/stylelib/presentation.mplstyle`` with the following::

   axes.titlesize : 24
   axes.labelsize : 20
   lines.linewidth : 3
   lines.markersize : 10
   xtick.labelsize : 16
   ytick.labelsize : 16

Then, when you want to adapt a plot designed for a paper to one that looks
good in a presentation, you can just add::

   >>> import matplotlib.pyplot as plt
   >>> plt.style.use('presentation')


Composing styles
================

Style sheets are designed to be composed together. So you can have a style
sheet that customizes colors and a separate style sheet that alters element
sizes for presentations. These styles can easily be combined by passing
a list of styles::

   >>> import matplotlib.pyplot as plt
   >>> plt.style.use(['dark_background', 'presentation'])

Note that styles further to the right will overwrite values that are already
defined by styles on the left.


Temporary styling
=================

If you only want to use a style for a specific block of code but don't want
to change the global styling, the style package provides a context manager
for limiting your changes to a specific scope. To isolate the your styling
changes, you can write something like the following::


   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>>
   >>> with plt.style.context(('dark_background')):
   >>>     plt.plot(np.sin(np.linspace(0, 2*np.pi)), 'r-o')
   >>>
   >>> # Some plotting code with the default style
   >>>
   >>> plt.show()


.. _matplotlibrc: http://matplotlib.sourceforge.net/users/customizing.html
.. _ggplot: http://had.co.nz/ggplot/
.. _R: http://www.r-project.org/
