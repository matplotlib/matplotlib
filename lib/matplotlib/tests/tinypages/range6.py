from matplotlib import pyplot as plt


def range4():
    """Never called if plot_directive works as expected."""
    raise NotImplementedError


def range6():
    """The function that should be executed."""
    plt.figure()
    plt.plot(range(6))
    plt.show()
