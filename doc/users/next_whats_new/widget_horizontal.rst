RadioButtons widget may now be laid out horizontally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.RadioButtons` widget's primary layout direction may now be specified with
the *layout_direction* keyword argument:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    fig = plt.figure(figsize=(4, 2))

    # Default orientation is vertical:
    rbv = RadioButtons(fig.add_axes((0.05, 0.6, 0.2, 0.35)),
                       ('Radio 1', 'Radio 2', 'Radio 3'),
                       layout_direction='vertical')

    # Alternatively, a horizontal orientation may be used:
    rbh = RadioButtons(fig.add_axes((0.3, 0.6, 0.6, 0.35)),
                       ('Radio 1', 'Radio 2', 'Radio 3'),
                       layout_direction='horizontal')
