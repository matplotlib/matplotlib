RadioButtons widget supports flexible layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.widgets.RadioButtons` widget now supports arranging buttons in different
layouts via the new *layout* parameter. You can arrange buttons vertically
(default), horizontally, or in a 2D grid by passing a ``(rows, cols)`` tuple.

The *active* parameter and the ``RadioButtons.index_selected`` attribute
continue to use a single integer index into the labels list. For grid layouts,
buttons are positioned left-to-right, top-to-bottom. The column positions are
automatically calculated based on the maximum text width in each column,
ensuring optimal spacing.

See :doc:`/gallery/widgets/radio_buttons_grid` for a ``(rows, cols)`` example.

.. plot::
    :include-source: true
    :alt: A sine wave plot with a 3x3 grid of radio buttons for selecting line color.

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import RadioButtons

    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*np.pi*t)

    fig, axes = plt.subplot_mosaic(
        [
            ['main'],
            ['.'],
            ['buttons'],
        ],
        height_ratios=[8, 0.4, 1],
    )

    line, = axes['main'].plot(t, s, lw=2, color='red')
    axes['main'].set_xlabel('Time (s)')
    axes['main'].set_ylabel('Amplitude')

    axes['buttons'].set_facecolor('lightgray')
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'black']
    radio = RadioButtons(axes['buttons'], colors, active=0, layout='horizontal')

    def color_func(label):
        line.set_color(label)
        fig.canvas.draw()

    radio.on_clicked(color_func)
    plt.show()
