Add step kwargs to fill_between
-------------------------------

Added ``step`` kwarg to `Axes.fill_between` to allow to fill between
lines drawn using the 'step' draw style.  The values of ``step`` match
those of the ``where`` kwarg of `Axes.step`.  The asymmetry of of the
kwargs names is not ideal, but `Axes.fill_between` already has a
``where`` kwarg.

This is particularly useful for plotting pre-binned histograms.

.. plot:: mpl_examples/api/filled_step.py
