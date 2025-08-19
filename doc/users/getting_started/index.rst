Getting started
===============

Installation quick-start
------------------------

.. grid:: 1 1 2 2

    .. grid-item::

        Install using `pip <https://pypi.org/project/matplotlib>`__:

        .. code-block:: bash

            pip install matplotlib

    .. grid-item::

        Install using `conda <https://docs.continuum.io/anaconda/>`__:

        .. code-block:: bash

            conda install -c conda-forge matplotlib

.. note::

   It is recommended to install Matplotlib inside a virtual environment
   (using ``python -m venv <envname>`` or ``conda create --name <envname>``),
   so that dependencies are managed cleanly and do not interfere with other projects.

Further details are available in the :doc:`Installation Guide </install/index>`.


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
   ax.plot(x, y, label="sin(x)")
   ax.set_title("A Simple Sine Curve")
   ax.set_xlabel("x values (radians)")
   ax.set_ylabel("sin(x)")
   ax.legend()
   plt.show()

If a plot does not show up please check :ref:`troubleshooting-faq`.


Where to go next
----------------

- Explore different :doc:`Plot types </plot_types/index>` available in Matplotlib.
- For a more detailed introduction, see the :ref:`Quick-start guide <quick_start>`.
- Browse the :doc:`Gallery </gallery/index>` for many examples you can copy and modify.
