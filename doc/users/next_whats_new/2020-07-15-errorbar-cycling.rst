``Axes.errorbar`` cycles non-color properties correctly
-------------------------------------------------------

Formerly, `.Axes.errorbar` incorrectly skipped the Axes property cycle if a
color was explicitly specified, even if the property cycler was for other
properties (such as line style). Now, `.Axes.errorbar` will advance the Axes
property cycle as done for `.Axes.plot`, i.e., as long as all properties in the
cycler are not explicitly passed.

For example, the following will cycle through the line styles:

.. plot::
    :include-source: True

    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    offsets = [0, 1]

    plt.rcParams['axes.prop_cycle'] = plt.cycler('linestyle', ['-', '--'])

    fig, ax = plt.subplots()
    for offset in offsets:
        ax.errorbar(x, y + offset, xerr=0.1, yerr=0.3, fmt='tab:blue')
