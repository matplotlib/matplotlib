API Changes for 3.6.1
=====================

Deprecations
------------

Colorbars for orphaned mappables are deprecated, but no longer raise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 3.6.0, Colorbars for mappables that do not have a parent Axes would
steal space from the current Axes. 3.6.0 raised an error on this, but without a
deprecation cycle. For 3.6.1 this is reverted; the current Axes is used, but a
deprecation warning is shown instead. In this undetermined case, users and
libraries should explicitly specify what Axes they want space to be stolen
from: ``fig.colorbar(mappable, ax=plt.gca())``.
