New StepPatch artist and a stairs method
----------------------------------------
`.pyplot.stairs` and the underlying artist `~.matplotlib.patches.StepPatch`
provide a cleaner interface for plotting stepwise constant functions for the
common case that you know the step edges. This superseeds many use cases of
`.pyplot.step`, for instance when plotting the output of `numpy.histogram`.

For both the artist and the function, the x-like edges input is one element
longer than the y-like values input

  .. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    h, edges = np.histogram(np.random.normal(5, 2, 5000),
                            bins=np.linspace(0,10,20))

    fig, ax = plt.subplots(constrained_layout=True)

    ax.stairs(h, edges)

    plt.show()

See :doc:`/gallery/lines_bars_and_markers/stairs_demo`
for examples.