Improved tick placement for ``symlog`` axes
-------------------------------------------

The placement of ticks for ``symlog`` axes has been improved. Ticks are now
placed identically to ``log`` axes in the logarithmic part with a reasonable
extension of this behavior to the linear part of the axis. Axes with too few
ticks or spurious ticks are avoided by the new implementation.

.. plot::
    :include-source: true

    # Show a logarithmic axis next to each symlog axis to illustrate the identical
    # ticking behavior in the logarithmic part of the symlog axis.

    fig = plt.figure()
    axs = fig.subplots(1, 4, gridspec_kw={'width_ratios': [0, 1, 0, 1]})

    npoints = 201
    for (sax, lax), (ymin, ymax) in zip([axs[[1, 0]], axs[[3, 2]]],
                                        [(-30, 200), (0.6, 6)]):
        # symlog plot
        sax.grid(which='major')
        sax.set_yscale('symlog')
        vals = np.linspace(ymin, ymax, npoints)
        sax.plot(vals, vals)
        # Calculate ymin for the log plot.
        trns = sax.yaxis.get_transform()
        mindec, maxdec = trns.transform_non_affine(np.array([ymin, ymax]))
        lymin = ymax * 10**(mindec - maxdec)
        # log plot
        lax.grid(which='major')
        lax.set_yscale('log')
        lax.xaxis.set_visible(False)
        vals = np.linspace(lymin, ymax, npoints)
        lax.plot(vals, vals)

    fig.tight_layout()
