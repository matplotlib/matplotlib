Subfigures are now added in row-major order
-------------------------------------------

``Figure.subfigures`` are now added in row-major order for API consistency.


.. plot::
    :include-source: true
    :alt: Example of creating 3 by 3 subfigures.

    import matplotlib.pyplot as plt

    fig = plt.figure()
    subfigs = fig.subfigures(3, 3)
    x = np.linspace(0, 10, 100)

    for i, sf in enumerate(fig.subfigs):
        ax = sf.subplots()
        ax.plot(x, np.sin(x + i), label=f'Subfigure {i+1}')
        sf.suptitle(f'Subfigure {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
