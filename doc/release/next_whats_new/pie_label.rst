Adding labels to pie chart wedges
---------------------------------

The new `~.Axes.pie_label` method adds a label to each wedge in a pie chart created with
`~.Axes.pie`.  It can take

* a list of strings, similar to the existing *labels* parameter of `~.Axes.pie`
* a format string similar to the existing *autopct* parameter of `~.Axes.pie` except
  that it uses the `str.format` method and it can handle absolute values as well as
  fractions/percentages

For more examples, see :doc:`/gallery/pie_and_polar_charts/pie_label`.

.. plot::
    :include-source: true
    :alt: A pie chart with three labels on each wedge, showing a food type, number, and fraction associated with the wedge.

    import matplotlib.pyplot as plt

    data = [36, 24, 8, 12]
    labels = ['spam', 'eggs', 'bacon', 'sausage']

    fig, ax = plt.subplots()
    pie = ax.pie(data)

    ax.pie_label(pie, labels, distance=1.1)
    ax.pie_label(pie, '{frac:.1%}', distance=0.7)
    ax.pie_label(pie, '{absval:d}', distance=0.4)
