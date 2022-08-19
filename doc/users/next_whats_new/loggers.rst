``set_loglevel`` can opt-out of manipulating logging handlers
-------------------------------------------------------------

It is now possible to configure the logging level of the Matplotilb standard
library logger without also implicitly installing a handler via both
`matplotlib.set_loglevel` and `matplotlib.pyplot.set_loglevel` ::

  mpl.set_loglevel('debug', ensure_handler=False)
  # or
  plt.set_loglevel('debug', ensure_handler=False)
