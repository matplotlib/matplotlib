
API Changes in 1.1.x
====================

* Added new :class:`matplotlib.sankey.Sankey` for generating Sankey diagrams.

* In :meth:`~matplotlib.pyplot.imshow`, setting *interpolation* to 'nearest'
  will now always mean that the nearest-neighbor interpolation is performed.
  If you want the no-op interpolation to be performed, choose 'none'.

* There were errors in how the tri-functions were handling input parameters
  that had to be fixed. If your tri-plots are not working correctly anymore,
  or you were working around apparent mistakes, please see issue #203 in the
  github tracker. When in doubt, use kwargs.

* The 'symlog' scale had some bad behavior in previous versions. This has now
  been fixed and users should now be able to use it without frustrations.
  The fixes did result in some minor changes in appearance for some users who
  may have been depending on the bad behavior.

* There is now a common set of markers for all plotting functions. Previously,
  some markers existed only for :meth:`~matplotlib.pyplot.scatter` or just for
  :meth:`~matplotlib.pyplot.plot`. This is now no longer the case. This merge
  did result in a conflict. The string 'd' now means "thin diamond" while
  'D' will mean "regular diamond".
