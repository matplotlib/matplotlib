#################
Nested page plots
#################

Plot 1 does not use context:

.. plot::

    plt.plot(range(10))
    plt.title('FIRST NESTED 1')
    a = 10

Plot 2 doesn't use context either; has length 6:

.. plot::

    plt.plot(range(6))
    plt.title('FIRST NESTED 2')


