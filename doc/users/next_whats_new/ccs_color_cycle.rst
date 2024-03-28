New more-accessible color cycle
-------------------------------

A new color cycle named 'ccs10' was added. This cycle was constructed using a
combination of algorithmically-enforced accessibility constraints, including
color-vision-deficiency modeling, and a machine-learning-based aesthetics model
developed from a crowdsourced color-preference survey. It aims to be both
generally pleasing aesthetically and colorblind accessible such that it could
serve as a default in the aim of universal design. For more details
see `Petroff, M. A.: "Accessible Color Sequences for Data Visualization"
<https://arxiv.org/abs/2107.02270>`_ and related `SciPy talk`_. A demonstration
is included in the style sheets reference_. To load this color cycle in place
of the default::

  import matplotlib.pyplot as plt
  plt.style.use('ccs10')

.. _reference: https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
.. _SciPy talk: https://www.youtube.com/watch?v=Gapv8wR5DYU
