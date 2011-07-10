from matplotlib.cbook import ls_mapper
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.tri.triangulation import Triangulation
import numpy as np

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
      triplot(x, y, mask=mask, ...)
      triplot(x, y, triangles, mask=mask, ...)

    in which case a Triangulation object will be created.  See
    :class:`~matplotlib.tri.Triangulation` for a explanation of these
    possibilities.

    The remaining args and kwargs are the same as for
    :meth:`~matplotlib.axes.Axes.plot`.

    **Example:**

        .. plot:: mpl_examples/pylab_examples/triplot_demo.py
    """
    import matplotlib.axes

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    x = tri.x
    y = tri.y
    edges = tri.edges

    # If draw both lines and markers at the same time, e.g.
    #     ax.plot(x[edges].T, y[edges].T, *args, **kwargs)
    # then the markers are drawn more than once which is incorrect if alpha<1.
    # Hence draw lines and markers separately.

    # Decode plot format string, e.g. 'ro-'
    fmt = ''
    if len(args) > 0:
        fmt = args[0]
    linestyle, marker, color = matplotlib.axes._process_plot_format(fmt)

    # Draw lines without markers, if lines are required.
    if linestyle is not None and linestyle is not 'None':
        kw = kwargs.copy()
        kw.pop('marker', None)     # Ignore marker if set.
        kw['linestyle'] = ls_mapper[linestyle]
        kw['edgecolor'] = color
        kw['facecolor'] = None

        vertices = np.column_stack((x[edges].flatten(), y[edges].flatten()))
        codes = ([Path.MOVETO] + [Path.LINETO])*len(edges)

        path = Path(vertices, codes)
        pathpatch = PathPatch(path, **kw)

        ax.add_patch(pathpatch)

    # Draw markers without lines.
    # Should avoid drawing markers for points that are not in any triangle?
    kwargs['linestyle'] = ''
    ax.plot(x, y, *args, **kwargs)
