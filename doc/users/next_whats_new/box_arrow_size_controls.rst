Arrow-style sub-classes of ``BoxStyle`` support arrow head resizing
-------------------------------------------------------------------

The new *head_width* and *head_angle* parameters to
`.BoxStyle.LArrow`, `.BoxStyle.RArrow` and `.BoxStyle.DArrow` allow for adjustment
of the size and aspect ratio of the arrow heads used.

By using negative angles (or corresponding reflex angles) for *head_angle*, arrows
with 'backwards' heads may be created.

.. plot::
    :include-source: false
    :alt: A plot containing two arrow-shaped text boxes. One arrow has a pentagonal 'road-sign' shape, and the other an inverted arrow head on each end.

    import numpy as np
    import matplotlib.pyplot as plt

    # Data for plotting; here, an intensity distribution for Fraunhofer diffraction
    # from 7 thin slits
    x_data = np.linspace(-3 * np.pi, 3 * np.pi, num=1000)
    I_data = (np.sin(x_data * 3.5) / np.sin(x_data / 2)) ** 2

    # Generate plot

    fig, ax = plt.subplots()
    plt.plot(x_data, I_data)

    plt.xlim(-3 * np.pi, 3 * np.pi)
    plt.ylim(0, 50)

    #
    # Annotate with boxed text in arrows
    #

    # head_width=1 gives 'road-sign' shape
    t1 = ax.text(-1, 35, "Primary maximum",
                ha="right", va="center", rotation=30, size=12,
                bbox=dict(boxstyle="rarrow,pad=0.3,head_width=1,head_angle=60",
                        fc="lightblue", ec="steelblue", lw=2))

    # Negative head_angle gives reversed arrow heads
    t2 = ax.text(np.pi, 30, "Lower intensity",
                ha="center", va="center", rotation=0, size=12,
                bbox=dict(boxstyle="darrow,pad=0.3,head_width=2,head_angle=-80",
                        fc="lightblue", ec="steelblue", lw=2))


    plt.show()
