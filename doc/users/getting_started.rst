Getting started
===============

Installation
------------

.. container:: twocol

    .. container::

        Install using pip:

        .. code-block:: bash

            pip install matplotlib

    .. container::

        Install using conda:

        .. code-block:: bash

            conda install matplotlib

Further details are available in the :doc:`Installation Guide </users/installing>`.  

Draw a first plot
-----------------

Here is a minimal example plot you can try out:

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

If a plot does not show up, we probably could not automatically find an appropriate backend; 
please check :ref:`troubleshooting-faq`.  

Where to go next
----------------

- Check out :doc:`Plot types </plot_types/index>` to get an overview of the
  types of plots you can create with Matplotlib.
- Learn Matplotlib from the ground up in the
  :doc:`Quick-start guide </tutorials/introductory/usage>`.
