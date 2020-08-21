New `~.matplotlib.patches.StepPatch` artist and a `.pyplot.histline` method
---------------------------------------------------------------------------
These take inputs of asymmetric lengths with y-like values and 
x-like edges, between which the values lie.

  .. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    h, bins = np.histogram(np.random.normal(5, 2, 5000),
                           bins=np.linspace(0,10,20))

    fig, ax = plt.subplots(constrained_layout=True)

    ax.histline(h, bins)

    plt.show()

See :doc:`/gallery/lines_bars_and_markers/histline_demo`
for examples.