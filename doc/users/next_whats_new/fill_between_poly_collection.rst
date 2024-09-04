``FillBetweenPolyCollection``
--------------------------------

The new class :class:`matplotlib.collections.FillBetweenPolyCollection` provides
the ``set_data`` method, enabling e.g. resampling
(:file:`galleries/event_handling/resample.html`).
:func:`matplotlib.axes.Axes.fill_between` and
:func:`matplotlib.axes.Axes.fill_betweenx` now return this new class.

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt

    t = np.linspace(0, 2, 9)
    f1 = t**2
    f2 = f1 + 0.2
    f3 = f2.copy()
    f3[6], f3[7], f3[8] = f3[8], f3[6], f3[7]

    fig, ax = plt.subplots(2)
    coll = ax.fill_between(t, f1, f3, step="pre")
    fig.savefig("before.png")

    coll.set_data(t, f1, f2, step="pre")
    fig.savefig("after.png")
