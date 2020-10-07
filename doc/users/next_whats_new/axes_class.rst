add_subplot/add_axes gained an *axes_class* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In particular, ``mpl_toolkits`` axes subclasses can now be idiomatically used
using e.g. ``fig.add_subplot(axes_class=mpl_toolkits.axislines.Axes)``
