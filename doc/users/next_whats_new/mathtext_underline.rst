Underlining text while using Mathtext
-------------------------------------

Mathtext now supports the ``\underline`` command.

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.text(0.4, 0.7, r'This is $\underline{underlined}$ text.')
    plt.text(0.4, 0.3, r'So is $\underline{\mathrm{this}}$.')
    plt.show()
