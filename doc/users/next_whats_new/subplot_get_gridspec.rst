Add ``ax.get_gridspec`` to `.SubplotBase`
-----------------------------------------

New method `.SubplotBase.get_gridspec` is added so that users can
easily get the gridspec that went into making an axes:

  .. code::

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2)
    gs = axs[0, -1].get_gridspec()

    # remove the last column
    for ax in axs[:,-1].flatten():
      ax.remove()

    # make a subplot in last column that spans rows.
    ax = fig.add_subplot(gs[:, -1])
    plt.show()
