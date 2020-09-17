New StepPatch artist and a stairs method
----------------------------------------
New `~.matplotlib.patches.StepPatch` artist and  `.pyplot.stairs` method.
For both the artist and the function, the x-like edges input is one 
longer than the y-like values input

  .. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    h, bins = np.histogram(np.random.normal(5, 2, 5000),
                           bins=np.linspace(0,10,20))

    fig, ax = plt.subplots(constrained_layout=True)

    ax.stairs(h, bins)

    plt.show()

See :doc:`/gallery/lines_bars_and_markers/stairs_demo`
for examples.