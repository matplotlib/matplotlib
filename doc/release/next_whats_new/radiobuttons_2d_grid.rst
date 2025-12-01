RadioButtons widget supports 2D grid layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.widgets.RadioButtons` widget now supports arranging buttons in a 2D grid
layout. Pass a list of lists of strings as the *labels* parameter to arrange
buttons in a grid where each inner list represents a row.

The *active* parameter and the ``RadioButtons.index_selected`` attribute
continue to use a single integer index into the flattened array, reading
left-to-right, top-to-bottom. The column positions are automatically calculated
based on the maximum text width in each column, ensuring optimal spacing.

See :doc:`/gallery/widgets/radio_buttons_grid` for a complete example.

.. plot::
    :include-source: true
    :alt: A sine wave plot with a 3x3 grid of radio buttons for selecting line color.

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import RadioButtons

    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*np.pi*t)

    fig, (ax_plot, ax_buttons) = plt.subplots(1, 2, figsize=(8, 4),
                                               width_ratios=[3, 1])

    line, = ax_plot.plot(t, s, lw=2, color='red')
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Amplitude')

    ax_buttons.set_facecolor('lightgray')
    ax_buttons.set_title('Line Color', fontsize=12, pad=10)

    colors = [
        ['red', 'orange', 'yellow'],
        ['green', 'blue', 'purple'],
        ['brown', 'pink', 'gray'],
    ]

    radio = RadioButtons(ax_buttons, colors, active=0)

    def color_func(label):
        line.set_color(label)
        fig.canvas.draw()

    radio.on_clicked(color_func)
    plt.show()
