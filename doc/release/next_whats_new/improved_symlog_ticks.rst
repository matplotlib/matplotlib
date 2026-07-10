Improved tick placement for ``symlog`` axes
-------------------------------------------

The placement of ticks for ``symlog`` axes has been improved. Ticks are now
placed identically to ``log`` axes in the logarithmic part with a reasonable
extension of this behavior to the linear part of the axis. Axes with too few
ticks or spurious ticks are avoided by the new implementation.

.. plot::
    :include-source: true

    fig, axs = plt.subplots(1, 2)
    x = np.arange(201)

    for ax, (ymin, ymax) in zip(axs, [(-30, 200), (0.6, 6)]):
        y = np.linspace(ymin, ymax, x.size)
        ax.set_yscale('symlog')
        ax.grid(which='major')
        ax.plot(x, y, '.')

    fig.tight_layout()
