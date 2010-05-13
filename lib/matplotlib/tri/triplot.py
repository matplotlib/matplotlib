from matplotlib.tri.triangulation import Triangulation

def triplot(ax, *args, **kwargs):
    """
    Draw a unstructured triangular grid as lines and/or markers to
    the :class:`~matplotlib.axes.Axes`.

    The triangulation to plot can be specified in one of two ways;
    either::

      triplot(triangulation, ...)

    where triangulation is a :class:`~matplotlib.tri.Triangulation`
    object, or

    ::

      triplot(x, y, ...)
      triplot(x, y, triangles, ...)
      triplot(x, y, triangles=triangles, ...)
      triplot(x, y, mask, ...)
      triplot(x, y, mask=mask, ...)
      triplot(x, y, triangles, mask, ...)
      triplot(x, y, triangles, mask=mask, ...)

    in which case a Triangulation object will be created.  See
    :class:`~matplotlib.tri.Triangulation` for a explanation of these
    possibilities.

    The remaining args and kwargs are the same as for
    :meth:`~matplotlib.axes.Axes.plot`.

    **Example:**

        .. plot:: mpl_examples/pylab_examples/triplot_demo.py
    """
    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    x = tri.x
    y = tri.y
    edges = tri.edges

    # If draw both lines and markers at the same time, e.g.
    #     ax.plot(x[edges].T, y[edges].T, *args, **kwargs)
    # then the markers are drawn more than once which is incorrect if alpha<1.
    # Hence draw of lines and markers separately.

    # Draw lines without markers.
    marker = kwargs.pop('marker', None)
    kwargs['marker'] = ''
    ax.plot(x[edges].T, y[edges].T, *args, **kwargs)
    if marker is None:
        kwargs.pop('marker')
    else:
        kwargs['marker'] = marker

    # Draw markers without lines.
    # Should avoid drawing markers for points that are not in any triangle?
    kwargs['linestyle'] = ''
    ax.plot(x, y, *args, **kwargs)
