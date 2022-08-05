import numpy as np

from matplotlib import _api
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.colors import Normalize
from matplotlib.tri.triangulation import Triangulation


def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None,
              vmax=None, shading='flat', facecolors=None, **kwargs):
    """
    Create a pseudocolor plot of an unstructured triangular grid.

    Call signatures::

      tripcolor(triangulation, C, *, ...)
      tripcolor(x, y, C, *, [triangles=triangles], [mask=mask], ...)

    The triangular grid can be specified either by passing a `.Triangulation`
    object as the first parameter, or by passing the points *x*, *y* and
    optionally the *triangles* and a *mask*. See `.Triangulation` for an
    explanation of these parameters.

    It is possible to pass the triangles positionally, i.e.
    ``tripcolor(x, y, triangles, C, ...)``. However, this is discouraged.
    For more clarity, pass *triangles* via keyword argument.

    If neither of *triangulation* or *triangles* are given, the triangulation
    is calculated on the fly. In this case, it does not make sense to provide
    colors at the triangle faces via *C* or *facecolors* because there are
    multiple possible triangulations for a group of points and you don't know
    which triangles will be constructed.

    Parameters
    ----------
    triangulation : `.Triangulation`
        An already created triangular grid.
    x, y, triangles, mask
        Parameters defining the triangular grid. See `.Triangulation`.
        This is mutually exclusive with specifying *triangulation*.
    C : array-like
        The color values, either for the points or for the triangles. Which one
        is automatically inferred from the length of *C*, i.e. does it match
        the number of points or the number of triangles. If there are the same
        number of points and triangles in the triangulation it is assumed that
        color values are defined at points; to force the use of color values at
        triangles use the keyword argument ``facecolors=C`` instead of just
        ``C``.
        This parameter is position-only.
    facecolors : array-like, optional
        Can be used alternatively to *C* to specify colors at the triangle
        faces. This parameter takes precedence over *C*.
    shading : {'flat', 'gouraud'}, default: 'flat'
        If  'flat' and the color values *C* are defined at points, the color
        values used for each triangle are from the mean C of the triangle's
        three points. If *shading* is 'gouraud' then color values must be
        defined at points.
    other_parameters
        All other parameters are the same as for `~.Axes.pcolor`.
    """
    _api.check_in_list(['flat', 'gouraud'], shading=shading)

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    # Parse the color to be in one of (the other variable will be None):
    # - facecolors: if specified at the triangle faces
    # - point_colors: if specified at the points
    if facecolors is not None:
        if args:
            _api.warn_external(
                "Positional parameter C has no effect when the keyword "
                "facecolors is given")
        point_colors = None
        if len(facecolors) != len(tri.triangles):
            raise ValueError("The length of facecolors must match the number "
                             "of triangles")
    else:
        # Color from positional parameter C
        if not args:
            raise TypeError(
                "tripcolor() missing 1 required positional argument: 'C'; or "
                "1 required keyword-only argument: 'facecolors'")
        elif len(args) > 1:
            _api.warn_deprecated(
                "3.6", message=f"Additional positional parameters "
                f"{args[1:]!r} are ignored; support for them is deprecated "
                f"since %(since)s and will be removed %(removal)s")
        C = np.asarray(args[0])
        if len(C) == len(tri.x):
            # having this before the len(tri.triangles) comparison gives
            # precedence to nodes if there are as many nodes as triangles
            point_colors = C
            facecolors = None
        elif len(C) == len(tri.triangles):
            point_colors = None
            facecolors = C
        else:
            raise ValueError('The length of C must match either the number '
                             'of points or the number of triangles')

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)

    edgecolors = 'none'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', edgecolors)

    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    if 'antialiaseds' not in kwargs and ec.lower() == "none":
        kwargs['antialiaseds'] = False

    _api.check_isinstance((Normalize, None), norm=norm)
    if shading == 'gouraud':
        if facecolors is not None:
            raise ValueError(
                "shading='gouraud' can only be used when the colors "
                "are specified at the points, not at the faces.")
        collection = TriMesh(tri, alpha=alpha, array=point_colors,
                             cmap=cmap, norm=norm, **kwargs)
    else:  # 'flat'
        # Vertices of triangles.
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # Color values.
        if facecolors is None:
            # One color per triangle, the mean of the 3 vertex color values.
            colors = point_colors[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # Remove color values of masked triangles.
            colors = facecolors[~tri.mask]
        else:
            colors = facecolors
        collection = PolyCollection(verts, alpha=alpha, array=colors,
                                    cmap=cmap, norm=norm, **kwargs)

    collection._scale_norm(norm, vmin, vmax)
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection
