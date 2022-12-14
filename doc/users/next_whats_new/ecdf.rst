``Axes.ecdf``
~~~~~~~~~~~~~
A new Axes method, `~.Axes.ecdf`, allows plotting empirical cumulative
distribution functions without any binning.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   fig, ax = plt.subplots()
   ax.ecdf(np.random.randn(100))
