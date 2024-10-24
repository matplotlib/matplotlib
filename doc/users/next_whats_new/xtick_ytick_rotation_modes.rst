``xtick`` and ``ytick`` rotation modes
--------------------------------------

A new feature has been added for handling rotation of xtick and ytick
labels more intuitively. The new rotation modes automatically adjusts the
alignment of rotated tick labels. This applies to tick labels on all four
sides of the plot (bottom, top, left, right), reducing the need for manual
adjustments when rotating labels.

.. plot::
    :include-source: true
    :alt: Example of rotated xtick and ytick labels.

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    s = range(5)
    ax1.set_xticks(s)
    ax1.set_xticklabels(['label'] * 5, rotation=-45, rotation_mode='xtick')
    ax1.set_yticks(s)
    ax1.set_yticklabels(['label'] * 5, rotation=45, rotation_mode='ytick')
    ax2.set_xticks(s)
    ax2.set_xticklabels(['label'] * 5, rotation=-45, rotation_mode='xtick')
    ax2.xaxis.tick_top()
    ax2.set_yticks(s)
    ax2.set_yticklabels(['label'] * 5, rotation=45, rotation_mode='ytick')
    ax2.yaxis.tick_right()

    plt.tight_layout()
    plt.show()
