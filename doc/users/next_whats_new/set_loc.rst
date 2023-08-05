Add a public method to modify the location of ``Legend``
--------------------------------------------------------

`~matplotlib.legend.Legend` locations now can be tweaked after they've been defined.

.. plot::
    :include-source: true

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = list(range(-100, 101))
    y = [i**2 for i in x]

    ax.plot(x, y, label="f(x)")
    ax.legend()
    ax.get_legend().set_loc("right")
    # Or
    # ax.get_legend().set(loc="right")

    plt.show()
