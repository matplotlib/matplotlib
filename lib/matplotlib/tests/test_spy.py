import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import check_figures_equal


@check_figures_equal()
def test_spy_box(fig_test, fig_ref):
    # setting up reference and test
    ax_test = fig_test.subplots(1, 3)
    ax_ref = fig_ref.subplots(1, 3)

    # plotting with spy
    ax_test[0].set_title("ones")
    ax_test[0].spy([[1, 1], [1, 1], ])
    ax_test[1].set_title("zeros")
    ax_test[1].spy([[0, 0], [0, 0], ])
    ax_test[2].set_title("mixed")
    ax_test[2].spy([[0, 1], [1, 0], ])

    # plotting with imshow
    ax_ref[0].set_title("ones")
    ax_ref[0].imshow([[1, 1], [1, 1], ], interpolation='nearest',
                        aspect='equal', origin='upper', cmap='Greys',
                        vmin=0, vmax=1)
    ax_ref[0].set_xlim(-0.5, 1.5)
    ax_ref[0].set_ylim(1.5, -0.5)
    ax_ref[0].xaxis.tick_top()
    ax_ref[0].title.set_y(1.05)
    ax_ref[0].xaxis.set_ticks_position('both')
    ax_ref[0].xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )
    ax_ref[0].yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )

    ax_ref[1].set_title("zeros")
    ax_ref[1].imshow([[0, 0], [0, 0], ], interpolation='nearest',
                        aspect='equal', origin='upper', cmap='Greys',
                        vmin=0, vmax=1)
    ax_ref[1].set_xlim(-0.5, 1.5)
    ax_ref[1].set_ylim(1.5, -0.5)
    ax_ref[1].xaxis.tick_top()
    ax_ref[1].title.set_y(1.05)
    ax_ref[1].xaxis.set_ticks_position('both')
    ax_ref[1].xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )
    ax_ref[1].yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )

    ax_ref[2].set_title("mixed")
    ax_ref[2].imshow([[0, 1], [1, 0], ], interpolation='nearest',
                        aspect='equal', origin='upper', cmap='Greys',
                        vmin=0, vmax=1)
    ax_ref[2].set_xlim(-0.5, 1.5)
    ax_ref[2].set_ylim(1.5, -0.5)
    ax_ref[2].xaxis.tick_top()
    ax_ref[2].title.set_y(1.05)
    ax_ref[2].xaxis.set_ticks_position('both')
    ax_ref[2].xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )
    ax_ref[2].yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
    )


