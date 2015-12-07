from matplotlib import pyplot as plt


def range4():
    '''This is never be called if plot_directive works as expected.'''
    raise NotImplementedError


def range6():
    plt.figure()
    plt.plot(range(6))
    plt.show()
