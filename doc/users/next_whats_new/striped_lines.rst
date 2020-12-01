Striped lines (experimental)
----------------------------

New *gapcolor* parameter enables the creation of striped lines.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(1., 3., 10)
    y = x**3

    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle='--', color='orange', gapcolor='blue',
            linewidth=3, label='a striped line')
    ax.legend()
    plt.show()
