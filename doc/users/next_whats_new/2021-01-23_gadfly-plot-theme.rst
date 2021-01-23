Gadfly Plot Theme
-------------------------

A new plotting theme for matplotlib based on the beautiful plots in the Gadfly plotting library for the Julia Programming language

.. plot::
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("gadfly")

    x = np.linspace(0, 10, 100)
    y = 2 * x
    plt.plot(x, y)