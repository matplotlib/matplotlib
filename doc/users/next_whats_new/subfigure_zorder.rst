Subfigures have now controllable zorders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, setting the zorder of a subfigure had no effect, and those were plotted on top of any figure-level artists (i.e for example on top of fig-level legends). Now, subfigures behave like any other artists, and their zorder can be controlled, with default a zorder of 0.

.. plot::
    :include-source: true
    :alt: Example on controlling the zorder of a subfigure

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(1, 10, 10)
    y1, y2 = x, -x
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2)
    for subfig in subfigs:
        axarr = subfig.subplots(2, 1)
        for ax in axarr.flatten():
            (l1,) = ax.plot(x, y1, label="line1")
            (l2,) = ax.plot(x, y2, label="line2")
    subfigs[0].set_zorder(6)
    l = fig.legend(handles=[l1, l2], loc="upper center", ncol=2)
