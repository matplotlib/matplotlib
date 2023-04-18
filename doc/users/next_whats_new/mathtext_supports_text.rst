``mathtext`` now supports ``\text``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``\text`` can be used to obtain upright text within an equation and to get a plain dash
(-).

.. plot::
    :include-source: true
    :alt: Illustration of the newly added \text command, showing that it renders as normal text, including spaces, despite being part of an equation. Also show that a dash is not rendered as a minus when part of a \text command.

    import matplotlib.pyplot as plt
    plt.text(0.1, 0.5, r"$a = \sin(\phi) \text{ such that } \phi = \frac{x}{y}$")
    plt.text(0.1, 0.3, r"$\text{dashes (-) are retained}$")
