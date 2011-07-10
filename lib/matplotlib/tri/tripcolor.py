from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.tri.triangulation import Triangulation
import numpy as np

def tripcolor(ax, *args, **kwargs):
    """
    Create a pseudocolor plot of an unstructured triangular grid to
    the :class:`~matplotlib.axes.Axes`.

    The triangulation can be specified in one of two ways; either::

      tripcolor(triangulation, ...)

    where triangulation is a :class:`~matplotlib.tri.Triangulation`
    object, or

    ::

      tripcolor(x, y, ...)
      tripcolor(x, y, triangles, ...)
      tripcolor(x, y, triangles=triangles, ...)
      tripcolor(x, y, mask=mask, ...)
      tripcolor(x, y, triangles, mask=mask, ...)

    in which case a Triangulation object will be created.  See
    :class:`~matplotlib.tri.Triangulation` for a explanation of these
    possibilities.

    The next argument must be *C*, the array of color values, one per
    point in the triangulation.  The colors used for each triangle
    are from the mean C of the triangle's three points.

    The remaining kwargs are the same as for
    :meth:`~matplotlib.axes.Axes.pcolor`.

    **Example:**

        .. plot:: mpl_examples/pylab_examples/tripcolor_demo.py
    """
    if not ax._hold: ax.cla()

    alpha = kwargs.pop('alpha', 1.0)
    norm = kwargs.pop('norm', None)
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    shading = kwargs.pop('shading', 'flat')

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    x = tri.x
    y = tri.y
    triangles = tri.get_masked_triangles()

    # Vertices of triangles.
    verts = np.concatenate((x[triangles][...,np.newaxis],
                            y[triangles][...,np.newaxis]), axis=2)

    C = np.asarray(args[0])
    if C.shape != x.shape:
        raise ValueError('C array must have same length as triangulation x and'
                         ' y arrays')

    # Color values, one per triangle, mean of the 3 vertex color values.
    C = C[triangles].mean(axis=1)

    if shading == 'faceted':
        edgecolors = (0,0,0,1),
        linewidths = (0.25,)
    else:
        edgecolors = 'face'
        linewidths = (1.0,)
    kwargs.setdefault('edgecolors', edgecolors)
    kwargs.setdefault('antialiaseds', (0,))
    kwargs.setdefault('linewidths', linewidths)

    collection = PolyCollection(verts, **kwargs)

    collection.set_alpha(alpha)
    collection.set_array(C)
    if norm is not None: assert(isinstance(norm, Normalize))
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)
    else:
        collection.autoscale_None()
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim( corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection
