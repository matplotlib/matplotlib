New *wedge_labels* parameter for pie
------------------------------------

`~.Axes.pie` now accepts a *wedge_labels* parameter which may be used to
annotate the wedges of the pie chart.  It can take

* a list of strings, similar to the existing *labels* parameter
* a format string in analogy to the existing *autopct* parameter except that it
  uses the `str.format` method, and it can handle absolute values as well as
  fractions/percentages

To add multiple labels per wedge, *wedge_labels* can take a sequence of any combination
of the above two options.

*wedge_labels* have accompanying *wedge_label_distance* and *rotate_wedge_labels*
parameters, to customise the position and rotation of the labels.

For examples, see :doc:`/gallery/pie_and_polar_charts/pie_features`.
