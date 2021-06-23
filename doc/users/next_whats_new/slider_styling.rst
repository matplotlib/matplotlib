Updated the appearance of Slider widgets
----------------------------------------

The appearance of `~.Slider` and `~.RangeSlider` widgets
were updated and given new styling parameters for the
added handles.

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    plt.figure(figsize=(4, 2))
    ax_old = plt.axes([0.2, 0.65, 0.65, 0.1])
    ax_new = plt.axes([0.2, 0.25, 0.65, 0.1])
    Slider(ax_new, "New", 0, 1)

    ax = ax_old
    valmin = 0
    valinit = 0.5
    ax.set_xlim([0, 1])
    ax_old.axvspan(valmin, valinit, 0, 1)
    ax.axvline(valinit, 0, 1, color="r", lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        -0.02,
        0.5,
        "Old",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="right",
    )

    ax.text(
        1.02,
        0.5,
        "0.5",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="left",
    )
