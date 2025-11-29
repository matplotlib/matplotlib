New *wedge_labels* parameter for pie
------------------------------------

`~.Axes.pie` now accepts a *wedge_labels* parameter as a shortcut to the
`~.Axes.pie_label` method. This may be used for simple annotation of the wedges
of the pie chart.  It can take

* a list of strings, similar to the existing *labels* parameter
* a format string similar to the existing *autopct* parameter except that it
  uses the `str.format` method, and it can handle absolute values as well as
  fractions/percentages

*wedge_labels* has an accompanying *wedge_label_distance* parameter, to control
the distance of the labels from the center of the pie.


.. plot::
    :include-source: true
    :alt: Two pie charts.  The chart on the left has labels 'foo' and 'bar' outside the wedges.  The chart on the right has labels '1' and '2' inside the wedges.

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(ncols=2, layout='constrained')

    ax1.pie([1, 2], wedge_labels=['foo', 'bar'], wedge_label_distance=1.1)
    ax2.pie([1, 2], wedge_labels='{absval:d}', wedge_label_distance=0.6)
