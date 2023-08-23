Remove inner ticks in ``label_outer()``
---------------------------------------
Up to now, ``label_outer()`` has only removed the ticklabels. The ticks lines
were left visible. This is now configurable through a new parameter
``label_outer(remove_inner_ticks=True)``.


.. plot::
   :include-source: true

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 100)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0, wspace=0))

    axs[0, 0].plot(x, np.sin(x))
    axs[0, 1].plot(x, np.cos(x))
    axs[1, 0].plot(x, -np.cos(x))
    axs[1, 1].plot(x, -np.sin(x))

    for ax in axs.flat:
        ax.grid(color='0.9')
        ax.label_outer(remove_inner_ticks=True)
