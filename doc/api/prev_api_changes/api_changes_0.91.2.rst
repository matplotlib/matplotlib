
Changes for 0.91.2
==================

* For ``csv2rec``, checkrows=0 is the new default indicating all rows
  will be checked for type inference

* A warning is issued when an image is drawn on log-scaled axes, since
  it will not log-scale the image data.

* Moved ``rec2gtk`` to ``matplotlib.toolkits.gtktools``

* Moved ``rec2excel`` to ``matplotlib.toolkits.exceltools``

* Removed, dead/experimental ExampleInfo, Namespace and Importer
  code from :mod:`matplotlib`
