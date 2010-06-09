.. _plotting-guide-legend:

************
Legend guide
************

Do not proceed unless you already have read :func:`~matplotlib.pyplot.legend` and
:class:`matplotlib.legend.Legend`!


What to be displayed
====================

The legend command has a following call signature::

      legend(*args, **kwargs)

If len(args) is 2, the first argument should be a list of artist to be
labeled, and the second argument should a list of string labels.  If
len(args) is 0, it automatically generate the legend from label
properties of the child artists by calling
:meth:`~matplotlib.axes.Axes.get_legend_handles_labels` method.
For example, *ax.legend()* is equivalent to::

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

The :meth:`~matplotlib.axes.Axes.get_legend_handles_labels` method
returns a tuple of two lists, i.e., list of artists and list of labels
(python string).  However, it does not return all of its child
artists. It returns all artists in *ax.lines* and *ax.patches* and
some artists in *ax.collection* which are instance of
:class:`~matplotlib.collections.LineCollection` or
:class:`~matplotlib.collections.RegularPolyCollection`.  The label
attributes (returned by get_label() method) of collected artists are
used as text labels. If label attribute is empty string or starts with
"_", that artist will be ignored.


 * Note that not all kind of artists are supported by the legend. The
   following is the list of artists that are currently supported.

   * :class:`~matplotlib.lines.Line2D`
   * :class:`~matplotlib.patches.Patch`
   * :class:`~matplotlib.collections.LineCollection`
   * :class:`~matplotlib.collections.RegularPolyCollection`

   Unfortunately, there is no easy workaround when you need legend for
   an artist not in the above list (You may use one of the supported
   artist as a proxy. See below), or customize it beyond what is
   supported by :class:`matplotlib.legend.Legend`.

 * Remember that some *pyplot* commands return artist not supported by
   legend, e.g., :func:`~matplotlib.pyplot.fill_between` returns
   :class:`~matplotlib.collections.PolyCollection` that is not
   supported. Or some return multiple artists. For example,
   :func:`~matplotlib.pyplot.plot` returns list of
   :class:`~matplotlib.lines.Line2D` instances, and
   :func:`~matplotlib.pyplot.errorbar` returns a length 3 tuple of
   :class:`~matplotlib.lines.Line2D` instances.

 * The legend does not care about the axes that given artists belongs,
   i.e., the artists may belong to other axes or even none.


Adjusting the Order of Legend items
-----------------------------------

When you want to customize the list of artists to be displayed in the
legend, or their order of appearance. There are a two options. First,
you can keep lists of artists and labels, and explicitly use these for
the first two argument of the legend call.::

  p1, = plot([1,2,3])
  p2, = plot([3,2,1])
  p3, = plot([2,3,1])
  legend([p2, p1], ["line 2", "line 1"])

Or you may use :meth:`~matplotlib.axes.Axes.get_legend_handles_labels`
to retrieve list of artist and labels and manipulate them before
feeding them to legend call.::

  ax = subplot(1,1,1)
  p1, = ax.plot([1,2,3], label="line 1")
  p2, = ax.plot([3,2,1], label="line 2")
  p3, = ax.plot([2,3,1], label="line 3")

  handles, labels = ax.get_legend_handles_labels()

  # reverse the order
  ax.legend(handles[::-1], labels[::-1])

  # or sort them by labels
  import operator
  hl = sorted(zip(handles, labels),
              key=operator.itemgetter(1))
  handles2, labels2 = zip(*hl)

  ax.legend(handles2, labels2)


Using Proxy Artist
------------------

When you want to display legend for an artist not supported by
matplotlib, you may use another artist as a proxy. For
example, you may create a proxy artist without adding it to the axes
(so the proxy artist will not be drawn in the main axes) and feed it
to the legend function.::

  p = Rectangle((0, 0), 1, 1, fc="r")
  legend([p], ["Red Rectangle"])


Multicolumn Legend
==================

By specifying the keyword argument *ncol*, you can have a multi-column
legend. Also, mode="expand" horizontally expand the legend to fill the
axes area. See `legend_demo3.py
<http://matplotlib.sourceforge.net/examples/pylab_examples/legend_demo3.html>`_
for example.


Legend location
===============

The location of the legend can be specified by the keyword argument
*loc*, either by string or a integer number.

=============  ======
 String        Number
=============  ======
 upper right    1
 upper left     2
 lower left     3
 lower right    4
 right          5
 center left    6
 center right   7
 lower center   8
 upper center   9
 center         10
=============  ======

By default, the legend will anchor to the bbox of the axes
(for legend) or the bbox of the figure (figlegend). You can specify
your own bbox using *bbox_to_anchor* argument. *bbox_to_anchor* can be an
instance of :class:`~matplotlib.transforms.BboxBase`, a tuple of 4
floats (x, y, width, height of the bbox), or a tuple of 2 floats (x, y
with width=height=0). Unless *bbox_transform* argument is given, the
coordinates (even for the bbox instance) are considered as normalized
axes coordinates.

For example, if you want your axes legend located at the figure corner
(instead of the axes corner)::

   l = legend(bbox_to_anchor=(0, 0, 1, 1), transform=gcf().transFigure)

Also, you can place above or outer right-hand side of the axes,

.. plot:: users/plotting/examples/simple_legend01.py
   :include-source:


Multiple Legend
===============

Sometime, you want to split the legend into multiple ones.::

  p1, = plot([1,2,3])
  p2, = plot([3,2,1])
  legend([p1], ["Test1"], loc=1)
  legend([p2], ["Test2"], loc=4)

However, the above code only shows the second legend. When the legend
command is called, a new legend instance is created and old ones are
removed from the axes. Thus, you need to manually add the removed
legend.

.. plot:: users/plotting/examples/simple_legend02.py
   :include-source:
