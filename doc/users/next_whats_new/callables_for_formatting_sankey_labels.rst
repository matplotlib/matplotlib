Support callable for formatting of Sankey labels
------------------------------------------------

The `format` parameter of `matplotlib.sankey.Sankey` can now accept callables.

This allows the use of an arbitrary function to label flows, for example allowing
the mapping of numbers to emoji.

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.sankey import Sankey
    import math


    def display_in_cats(values, min_cats, max_cats):
        def display_in_cat_scale(value):
            max_value = max(values, key=abs)
            number_cats_to_show = \
                max(min_cats, math.floor(abs(value) / max_value * max_cats))
            return str(number_cats_to_show * '🐱')

        return display_in_cat_scale


    flows = [35, 15, 40, -20, -15, -5, -40, -10]
    orientations = [-1, 1, 0, 1, 1, 1, -1, -1]

    # Cats are good, we want a strictly positive number of them
    min_cats = 1
    # More than four cats might be too much for some people
    max_cats = 4

    cats_format = display_in_cats(flows, min_cats, max_cats)

    sankey = Sankey(flows=flows, orientations=orientations, format=cats_format,
                    offset=.1, head_angle=180, shoulder=0, scale=.010)

    diagrams = sankey.finish()

    diagrams[0].texts[2].set_text('')

    plt.title(f'Sankey flows measured in cats \n'
              f'🐱 = {max(flows, key=abs) / max_cats}')

    plt.show()
