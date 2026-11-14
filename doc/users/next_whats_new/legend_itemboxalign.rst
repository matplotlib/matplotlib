Adding itembox alignment option for legends
---------------------------------------------------------

`~.Legend` previously always aligns items using the "baseline" option, which results in
the appearance of vertical centering of the artist and label for multi-line labels.
This is sometimes hard to read. The introduction of the `itemboxalign` parameter allows
the user to change this behavior and choose a different desired vertical alignment.

.. plot::
    :include-source: true
    :alt: A legend with artist and label aligned to 'top' rather than 'baseline'

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    ax[0].plot([5, 2, 8], label='long\nlabel')
    ax[0].plot([4, 9, 1], label='another\nlong\nlabel')
    ax[0].legend(title="align=baseline (default)")

    ax[1].plot([5, 2, 8], label='long\nlabel')
    ax[1].plot([4, 9, 1], label='another\nlong\nlabel')
    ax[1].legend(title="align=top", itemboxalign='top')

    plt.show()
