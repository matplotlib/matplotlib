``FillBetweenPolyCollection``
-----------------------------

The new class :class:`matplotlib.collections.FillBetweenPolyCollection` provides
the ``set_data`` method, enabling e.g. resampling
(:file:`galleries/event_handling/resample.html`).
:func:`matplotlib.axes.Axes.fill_between` and
:func:`matplotlib.axes.Axes.fill_betweenx` now return this new class.

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt

    t = np.linspace(0, 1)

    fig, ax = plt.subplots()
    coll = ax.fill_between(t, -t**2, t**2)
    fig.savefig("before.png")

    coll.set_data(t, -t**4, t**4)
    fig.savefig("after.png")
