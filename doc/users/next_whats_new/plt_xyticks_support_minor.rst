``plt.xticks`` and ``plt.yticks`` support *minor* keyword argument
------------------------------------------------------------------

It is now possible to set or get minor ticks using `.pyplot.xticks` and
`.pyplot.yticks` by setting ``minor=True``.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1, 2, 3, 3.5], [2, 1, 0, -0.5])
    plt.xticks([1, 2, 3], ["One", "Zwei", "Trois"])
    plt.xticks([1.414, 2.5, 3.142],
               [r"$\sqrt{2}$", r"$\frac{5}{2}$", r"$\pi$"], minor=True)
