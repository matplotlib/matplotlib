Plot directive caption tests
============================

Inline plot with no caption:

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   f = 3
   t = np.linspace(0, 1, 100)
   s = np.sin(2 * np.pi * f * t)
   plt.plot(t, s)

Inline plot with caption:

.. plot::
   :caption: Caption for inline plot.

   import matplotlib.pyplot as plt
   import numpy as np
   f = 3
   t = np.linspace(0, 1, 100)
   s = np.sin(2 * np.pi * f * t)
   plt.plot(t, s)

Included file with no caption:

.. plot:: test_plot.py

Included file with caption in the directive content:

.. plot:: test_plot.py

   This is a caption in the content.

Included file with caption option:

.. plot:: test_plot.py
   :caption: This is a caption in the options.

If both content and options have a caption, the one in the content should prevail:

.. plot:: test_plot.py
   :caption: This should be ignored.

   The content caption should be used instead.
