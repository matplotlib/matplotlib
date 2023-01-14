
####################
Adding text to plots
####################

.. plot::

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)

    # Set titles for the figure and the subplot respectively
    fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    ax.set_title('axes title')

    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    # Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
    ax.axis([0, 10, 0, 10])

    ax.text(3, 8, 'boxed italics text in data coords', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

    ax.text(3, 2, 'Unicode: Institut für Festkörperphysik')

    ax.text(0.95, 0.01, 'colored text in axes coords',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)

    ax.plot([2], [1], 'o')
    ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()

.. toctree::
    :maxdepth: 1

    ../../../tutorials/text/text_intro.rst
    ../../../tutorials/intermediate/legend_guide.rst
    ../../../tutorials/text/annotations.rst
    ../../../tutorials/text/text_props.rst
    ../../../tutorials/text/mathtext.rst
    ../../../tutorials/text/usetex.rst
    ../fonts.rst
