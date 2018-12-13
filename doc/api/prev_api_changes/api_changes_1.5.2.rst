Changes in 1.5.2
================


Default Behavior Changes
------------------------

Changed default ``autorange`` behavior in boxplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prior to v1.5.2, the whiskers of boxplots would extend to the minimum
and maximum values if the quartiles were all equal (i.e., Q1 = median
= Q3). This behavior has been disabled by default to restore consistency
with other plotting packages.

To restore the old behavior, simply set ``autorange=True`` when
calling ``plt.boxplot``.
