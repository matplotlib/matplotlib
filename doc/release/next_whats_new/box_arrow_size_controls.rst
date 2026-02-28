Arrow-style sub-classes of ``BoxStyle`` support arrow head resizing
-------------------------------------------------------------------

The new *head_width* and *head_angle* parameters to
`.BoxStyle.LArrow`, `.BoxStyle.RArrow` and `.BoxStyle.DArrow` allow for adjustment
of the size and aspect ratio of the arrow heads used.

To give a consistent appearance across all parameter values, the
default head position (where the head starts relative to text) is
slightly changed compared to the previous hard-coded position.

By using negative angles (or corresponding reflex angles) for *head_angle*, arrows
with 'backwards' heads may be created.

.. plot::
    :include-source: true
    :alt: Six arrow-shaped text boxes, all containing the text "Arrow". The top left arrow has a shorter head than default, while the top right arrow a longer head. The centre left double arrow has a "road-sign" shape (head as wide as the arrow tail), while the centre right arrow has a "backwards" head. The bottom left arrow has two heads which are larger than default, and the bottom right arrow has a head narrower than its tail.

    import matplotlib.pyplot as plt

    plt.text(0.2, 0.8, "Arrow", ha='center', size=16, bbox=dict(boxstyle="larrow, pad=0.3, head_angle=150"))
    plt.text(0.7, 0.8, "Arrow", ha='center', size=16, bbox=dict(boxstyle="rarrow, pad=0.3, head_angle=30"))
    plt.text(0.2, 0.2, "Arrow", ha='center', size=16, bbox=dict(boxstyle="darrow, pad=0.3, head_width=3"))
    plt.text(0.7, 0.2, "Arrow", ha='center', size=16, bbox=dict(boxstyle="larrow, pad=0.3, head_width=0.5"))
    plt.text(0.2, 0.5, "Arrow", ha='center', size=16, bbox=dict(boxstyle="darrow, pad=0.3, head_width=1, head_angle=60"))
    plt.text(0.7, 0.5, "Arrow", ha='center', size=16, bbox=dict(boxstyle="rarrow, pad=0.3, head_width=2, head_angle=-90"))

    plt.show()
