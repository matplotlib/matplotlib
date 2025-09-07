Six and eight color Petroff color cycles
----------------------------------------

The six and eight color accessible Petroff color cycles are named 'petroff6' and
'petroff8'.
They compliment the existing 'petroff10' color cycle, added in `Matplotlib 3.10.0`_

For more details see
`Petroff, M. A.: "Accessible Color Sequences for Data Visualization"
<https://arxiv.org/abs/2107.02270>`_.
To load the 'petroff6' color cycle in place of the default::

  import matplotlib.pyplot as plt
  plt.style.use('petroff6')

or to load the 'petroff8' color cycle::

  import matplotlib.pyplot as plt
  plt.style.use('petroff8')

.. _Matplotlib 3.10.0: https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.10.0.html#new-more-accessible-color-cycle
