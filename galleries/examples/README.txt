
.. _examples-index:

.. _gallery:

========
Examples
========
For an overview of the plotting methods we provide, see :ref:`plot_types`

This page contains example plots. Click on any image to see the full image
and source code.

For longer tutorials, see our :ref:`tutorials page <tutorials>`.
You can also find :ref:`external resources <resources-index>` and
a :ref:`FAQ <faq-index>` in our :ref:`user guide <users-guide-index>`.


.. admonition:: Tagging!

    You can also browse the example gallery by :ref:`tags <tagoverview>`.


Live example (experimental)
===========================

Try Matplotlib directly in this documentation (press :kbd:`shift` + :kbd:`Enter` to execute code)!

.. rstcheck: ignore-directives=replite
.. replite::
   :kernel: xeus-python
   :height: 600px
   :prompt: Try Matplotlib!
   :execute: False

   %matplotlib inline

   import matplotlib.pyplot as plt
   import numpy as np

   fig = plt.figure()
   plt.plot(np.sin(np.linspace(0, 20, 100)))
   plt.show();

Alternatively, you can try the gallery examples below in `our JupyterLite deployment <./../lite/lab>`__.
