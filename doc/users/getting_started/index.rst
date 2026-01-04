Getting started
===============

Installation quick-start
------------------------

For up-to-date installation instructions, please refer to the official
Matplotlib installation guide:

https://matplotlib.org/stable/#install

Draw a first plot
-----------------

Here is a minimal example plot:

.. plot::
   :include-source:
   :align: center

   import matplotlib.pyplot as plt
   import numpy as np

   x = np.linspace(0, 2 * np.pi, 200)
   y = np.sin(x)

   fig, ax = plt.subplots()
   ax.plot(x, y)
   plt.show()

If a plot does not show up please check :ref:`troubleshooting-faq`.

Where to go next
----------------

- Check out :doc:`Plot types </plot_types/index>` to get an overview of the
  types of plots you can create with Matplotlib.
- Learn Matplotlib from the ground up in the :ref:`Quick-start guide
  <quick_start>`.
