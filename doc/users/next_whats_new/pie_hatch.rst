``hatch`` parameter for pie
-------------------------------------------

`~matplotlib.axes.Axes.pie` now accepts a *hatch* keyword that takes as input
a hatch or list of hatches:

.. plot::
    :include-source: true
    :alt: Two pie charts, identified as ax1 and ax2, both have a small blue slice, a medium orange slice, and a large green slice. ax1 has a dot hatching on the small slice, a small open circle hatching on the medium slice, and a large open circle hatching on the large slice. ax2 has the same large open circle with a dot hatch on every slice.

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    x = [10, 30, 60]

    ax1.pie(x, hatch=['.', 'o', 'O'])
    ax2.pie(x, hatch='.O')

    ax1.set_title("hatch=['.', 'o', 'O']")
    ax2.set_title("hatch='.O'")
