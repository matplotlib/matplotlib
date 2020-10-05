New CenteredNorm for symmetrical data around a center
-----------------------------------------------------
In cases where data is symmetrical around a center, for example, positive and
negative anomalies around a center zero, `~.matplotlib.colors.CenteredNorm`
is a new norm that automatically creates a symmetrical mapping around the
center. This norm is well suited to be combined with a divergent colormap which
uses an unsaturated color in its center.

  .. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import CenteredNorm

    np.random.seed(20201004)
    data = np.random.normal(size=(3, 4), loc=1)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(data, cmap=plt.get_cmap('RdGy'), norm=CenteredNorm())
    fig.colorbar(pc)
    ax.set_title('data centered around zero')

    # add text annotation
    for irow, data_row in enumerate(data):
        for icol, val in enumerate(data_row):
            ax.text(icol + 0.5, irow + 0.5, f'{val:.2f}', color='C0',
                    size=16, va='center', ha='center')
    plt.show()

If the center of symmetry is different from 0, it can be set with the *vcenter*
argument. To manually set the range of `~.matplotlib.colors.CenteredNorm`, use
the *halfrange* argument.

See :doc:`/tutorials/colors/colormapnorms` for an example and more details
about data normalization.
