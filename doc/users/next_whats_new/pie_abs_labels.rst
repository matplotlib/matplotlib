Label pie charts with absolute input values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pie charts may now be automatically labelled with the original input values
using the new *absolutefmt* parameter in `~.Axes.pie`.

.. plot::
    :include-source: true
    :alt: Pie chart with each wedge labelled with its input value

    import matplotlib.pyplot as plt

    x = [4, 2, 1]

    fig, ax = plt.subplots()
    ax.pie(x, absolutefmt='%d')

    plt.show()
