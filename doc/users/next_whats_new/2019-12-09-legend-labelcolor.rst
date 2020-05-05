Text color for legend labels
----------------------------

The text color of legend labels can now be set by passing a parameter
``labelcolor`` to `~.axes.Axes.legend`. The ``labelcolor`` keyword can be:

* A single color (either a string or RGBA tuple), which adjusts the text color
  of all the labels.
* A list or tuple, allowing the text color of each label to be set
  individually.
* ``linecolor``, which sets the text color of each label to match the
  corresponding line color.
* ``markerfacecolor``, which sets the text color of each label to match the
  corresponding marker face color.
* ``markeredgecolor``,  which sets the text color of each label to match the
  corresponding marker edge color.
