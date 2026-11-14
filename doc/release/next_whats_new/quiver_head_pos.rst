Position of the arrowhead in ``quiver``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.Axes.quiver` has a new *head_pos* parameter that sets where the arrowhead
sits along the shaft. It can be 'tail', 'middle' or 'tip' (the default), or a
number between 0 (at the tail) and 1 (at the tip).

.. plot::
    :include-source: true
    :alt: Three rows of arrows, with the arrowhead at the tail, the middle and the tip of the shaft.

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
    head_positions = ['tip', 'middle', 'tail']
    for y, head_pos in enumerate(head_positions):
        ax.quiver([0, 1, 2], [y] * 3, 1, 0.3, head_pos=head_pos,
                  angles='xy', scale_units='xy', scale=1.5)
    ax.set(xlim=(-0.3, 3), ylim=(-0.5, 2.7), xticks=[], yticks=[0, 1, 2],
           yticklabels=[f'head_pos={p!r}' for p in head_positions])
