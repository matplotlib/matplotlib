path.effects rcParam can be set in stylesheet and new xkcd stylesheet
---------------------------------------------------------------------

Can now set the ``path.effects`` :ref:`rcParam in a style sheet <customizing>`
using a list of ``(patheffects function name, {**kwargs})``::

    path.effects: ('Normal', ), ('Stroke', {'offset': (1, 2)}), ('withStroke', {'linewidth': 4, 'foreground': 'w'})


This feature means that the xkcd style can be used like any other stylesheet:

.. plot::
    :include-source: true
    :alt: plot where graph and text appear in a hand drawn comic like style

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2* np.pi, 100)
    y = np.sin(x)

    with plt.style.context('xkcd'):

        fig, ax = plt.subplots()
        ax.set_title("sine curve")
        ax.plot(x, y)
