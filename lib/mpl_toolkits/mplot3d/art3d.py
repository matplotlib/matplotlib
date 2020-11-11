# art3d.py, original mplot3d version by John Porter
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>
# Minor additions by Ben Axelrod <baxelrod@coroware.com>

"""
Module containing 3D artist code and functions to convert 2D
artists into 3D versions which can be added to an Axes3D.
"""

import math

import numpy as np

from matplotlib import (
    artist, colors as mcolors, lines, text as mtext, path as mpath)
from matplotlib.collections import (
    LineCollection, PolyCollection, PatchCollection, PathCollection)
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d


def _norm_angle(a):
    """Return the given angle normalized to -180 < *a* <= 180 degrees."""
    a = (a + 360) % 360
    if a > 180:
        a = a - 360
    return a


def _norm_text_angle(a):
    """Return the given angle normalized to -90 < *a* <= 90 degrees."""
    a = (a + 180) % 180
    if a > 90:
        a = a - 180
    return a


def get_dir_vector(zdir):
    """
    Return a direction vector.

    Parameters
    ----------
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction. Possible values are:
        - 'x': equivalent to (1, 0, 0)
        - 'y': equivalent to (0, 1, 0)
        - 'z': equivalent to (0, 0, 1)
        - *None*: equivalent to (0, 0, 0)
        - an iterable (x, y, z) is returned unchanged.

    Returns
    -------
    x, y, z : array-like
        The direction vector. This is either a numpy.array or *zdir* itself if
        *zdir* is already a length-3 iterable.

    """
    if zdir == 'x':
        return np.array((1, 0, 0))
    elif zdir == 'y':
        return np.array((0, 1, 0))
    elif zdir == 'z':
        return np.array((0, 0, 1))
    elif zdir is None:
        return np.array((0, 0, 0))
    elif np.iterable(zdir) and len(zdir) == 3:
        return zdir
    else:
        raise ValueError("'x', 'y', 'z', None or vector of length 3 expected")


class Text3D(mtext.Text):
    """
    Text object with 3D position and direction.

    Parameters
    ----------
    x, y, z
        The position of the text.
    text : str
        The text string to display.
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction of the text. See `.get_dir_vector` for a description of
        the values.

    Other Parameters
    ----------------
    **kwargs
         All other parameters are passed on to `~matplotlib.text.Text`.
   """

    def __init__(self, x=0, y=0, z=0, text='', zdir='z', **kwargs):
        mtext.Text.__init__(self, x, y, text, **kwargs)
        self.set_3d_properties(z, zdir)

    def set_3d_properties(self, z=0, zdir='z'):
        x, y = self.get_position()
        self._position3d = np.array((x, y, z))
        self._dir_vec = get_dir_vector(zdir)
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        proj = proj3d.proj_trans_points(
            [self._position3d, self._position3d + self._dir_vec], renderer.M)
        dx = proj[0][1] - proj[0][0]
        dy = proj[1][1] - proj[1][0]
        angle = math.degrees(math.atan2(dy, dx))
        self.set_position((proj[0][0], proj[1][0]))
        self.set_rotation(_norm_text_angle(angle))
        mtext.Text.draw(self, renderer)
        self.stale = False

    def get_tightbbox(self, renderer):
        # Overwriting the 2d Text behavior which is not valid for 3d.
        # For now, just return None to exclude from layout calculation.
        return None


def text_2d_to_3d(obj, z=0, zdir='z'):
    """Convert a Text to a Text3D object."""
    obj.__class__ = Text3D
    obj.set_3d_properties(z, zdir)


class Line3D(lines.Line2D):
    """
    3D line object.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Keyword arguments are passed onto :func:`~matplotlib.lines.Line2D`.
        """
        lines.Line2D.__init__(self, [], [], *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_3d_properties(self, zs=0, zdir='z'):
        xs = self.get_xdata()
        ys = self.get_ydata()
        zs = np.broadcast_to(zs, xs.shape)
        self._verts3d = juggle_axes(xs, ys, zs, zdir)
        self.stale = True

    def set_data_3d(self, *args):
        """
        Set the x, y and z data

        Parameters
        ----------
        x : array-like
            The x-data to be plotted.
        y : array-like
            The y-data to be plotted.
        z : array-like
            The z-data to be plotted.

        Notes
        -----
        Accepts x, y, z arguments or a single array-like (x, y, z)
        """
        if len(args) == 1:
            self._verts3d = args[0]
        else:
            self._verts3d = args
        self.stale = True

    def get_data_3d(self):
        """
        Get the current data

        Returns
        -------
        verts3d : length-3 tuple or array-like
            The current data as a tuple or array-like.
        """
        return self._verts3d

    @artist.allow_rasterization
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_data(xs, ys)
        lines.Line2D.draw(self, renderer)
        self.stale = False


def line_2d_to_3d(line, zs=0, zdir='z'):
    """Convert a 2D line to 3D."""

    line.__class__ = Line3D
    line.set_3d_properties(zs, zdir)


def _path_to_3d_segment(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment."""

    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg = [(x, y, z) for (((x, y), code), z) in zip(pathsegs, zs)]
    seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    return seg3d


def _paths_to_3d_segments(paths, zs=0, zdir='z'):
    """Convert paths from a collection object to 3D segments."""

    zs = np.broadcast_to(zs, len(paths))
    segs = [_path_to_3d_segment(path, pathz, zdir)
            for path, pathz in zip(paths, zs)]
    return segs


def _path_to_3d_segment_with_codes(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment with path codes."""

    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg_codes = [((x, y, z), code) for ((x, y), code), z in zip(pathsegs, zs)]
    if seg_codes:
        seg, codes = zip(*seg_codes)
        seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    else:
        seg3d = []
        codes = []
    return seg3d, list(codes)


def _paths_to_3d_segments_with_codes(paths, zs=0, zdir='z'):
    """
    Convert paths from a collection object to 3D segments with path codes.
    """

    zs = np.broadcast_to(zs, len(paths))
    segments_codes = [_path_to_3d_segment_with_codes(path, pathz, zdir)
                      for path, pathz in zip(paths, zs)]
    if segments_codes:
        segments, codes = zip(*segments_codes)
    else:
        segments, codes = [], []
    return list(segments), list(codes)


class Line3DCollection(LineCollection):
    """
    A collection of 3D lines.
    """

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_segments(self, segments):
        """
        Set 3D segments.
        """
        self._segments3d = segments
        LineCollection.set_segments(self, [])

    def do_3d_projection(self, renderer):
        """
        Project the points according to renderer matrix.
        """
        # see _update_scalarmappable docstring for why this must be here
        _update_scalarmappable(self)
        xyslist = [
            proj3d.proj_trans_points(points, renderer.M) for points in
            self._segments3d]
        segments_2d = [np.column_stack([xs, ys]) for xs, ys, zs in xyslist]
        LineCollection.set_segments(self, segments_2d)

        # FIXME
        minz = 1e9
        for xs, ys, zs in xyslist:
            minz = min(minz, min(zs))
        return minz

    @artist.allow_rasterization
    def draw(self, renderer, project=False):
        if project:
            self.do_3d_projection(renderer)
        LineCollection.draw(self, renderer)


def line_collection_2d_to_3d(col, zs=0, zdir='z'):
    """Convert a LineCollection to a Line3DCollection object."""
    segments3d = _paths_to_3d_segments(col.get_paths(), zs, zdir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)


class Patch3D(Patch):
    """
    3D patch object.
    """

    def __init__(self, *args, zs=(), zdir='z', **kwargs):
        Patch.__init__(self, *args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_3d_properties(self, verts, zs=0, zdir='z'):
        zs = np.broadcast_to(zs, len(verts))
        self._segment3d = [juggle_axes(x, y, z, zdir)
                           for ((x, y), z) in zip(verts, zs)]
        self._facecolor3d = Patch.get_facecolor(self)

    def get_path(self):
        return self._path2d

    def get_facecolor(self):
        return self._facecolor2d

    def do_3d_projection(self, renderer):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, renderer.M)
        self._path2d = mpath.Path(np.column_stack([vxs, vys]))
        # FIXME: coloring
        self._facecolor2d = self._facecolor3d
        return min(vzs)


class PathPatch3D(Patch3D):
    """
    3D PathPatch object.
    """

    def __init__(self, path, *, zs=(), zdir='z', **kwargs):
        Patch.__init__(self, **kwargs)
        self.set_3d_properties(path, zs, zdir)

    def set_3d_properties(self, path, zs=0, zdir='z'):
        Patch3D.set_3d_properties(self, path.vertices, zs=zs, zdir=zdir)
        self._code3d = path.codes

    def do_3d_projection(self, renderer):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, renderer.M)
        self._path2d = mpath.Path(np.column_stack([vxs, vys]), self._code3d)
        # FIXME: coloring
        self._facecolor2d = self._facecolor3d
        return min(vzs)


def _get_patch_verts(patch):
    """Return a list of vertices for the path of a patch."""
    trans = patch.get_patch_transform()
    path = patch.get_path()
    polygons = path.to_polygons(trans)
    if len(polygons):
        return polygons[0]
    else:
        return []


def patch_2d_to_3d(patch, z=0, zdir='z'):
    """Convert a Patch to a Patch3D object."""
    verts = _get_patch_verts(patch)
    patch.__class__ = Patch3D
    patch.set_3d_properties(verts, z, zdir)


def pathpatch_2d_to_3d(pathpatch, z=0, zdir='z'):
    """Convert a PathPatch to a PathPatch3D object."""
    path = pathpatch.get_path()
    trans = pathpatch.get_patch_transform()

    mpath = trans.transform_path(path)
    pathpatch.__class__ = PathPatch3D
    pathpatch.set_3d_properties(mpath, z, zdir)


class Patch3DCollection(PatchCollection):
    """
    A collection of 3D patches.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        """
        Create a collection of flat 3D patches with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of patches in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PatchCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument "depthshade" is available to
        indicate whether or not to shade the patches in order to
        give the appearance of depth (default is *True*).
        This is typically desired in scatter plots.
        """
        self._depthshade = depthshade
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir):
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._facecolor3d = self.get_facecolor()
        self._edgecolor3d = self.get_edgecolor()
        self.stale = True

    def do_3d_projection(self, renderer):
        # see _update_scalarmappable docstring for why this must be here
        _update_scalarmappable(self)
        xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, renderer.M)

        fcs = (_zalpha(self._facecolor3d, vzs) if self._depthshade else
               self._facecolor3d)
        fcs = mcolors.to_rgba_array(fcs, self._alpha)
        self.set_facecolors(fcs)

        ecs = (_zalpha(self._edgecolor3d, vzs) if self._depthshade else
               self._edgecolor3d)
        ecs = mcolors.to_rgba_array(ecs, self._alpha)
        self.set_edgecolors(ecs)
        PatchCollection.set_offsets(self, np.column_stack([vxs, vys]))

        if vzs.size > 0:
            return min(vzs)
        else:
            return np.nan


class Path3DCollection(PathCollection):
    """
    A collection of 3D paths.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        """
        Create a collection of flat 3D paths with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of paths in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PathCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument "depthshade" is available to
        indicate whether or not to shade the patches in order to
        give the appearance of depth (default is *True*).
        This is typically desired in scatter plots.
        """
        self._depthshade = depthshade
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir):
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._facecolor3d = self.get_facecolor()
        self._edgecolor3d = self.get_edgecolor()
        self._sizes3d = self.get_sizes()
        self._linewidth3d = self.get_linewidth()
        self.stale = True

    def do_3d_projection(self, renderer):
        # see _update_scalarmappable docstring for why this must be here
        _update_scalarmappable(self)
        xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, renderer.M)

        fcs = (_zalpha(self._facecolor3d, vzs) if self._depthshade else
               self._facecolor3d)
        ecs = (_zalpha(self._edgecolor3d, vzs) if self._depthshade else
               self._edgecolor3d)
        sizes = self._sizes3d
        lws = self._linewidth3d

        # Sort the points based on z coordinates
        # Performance optimization: Create a sorted index array and reorder
        # points and point properties according to the index array
        z_markers_idx = np.argsort(vzs)[::-1]

        # Re-order items
        vzs = vzs[z_markers_idx]
        vxs = vxs[z_markers_idx]
        vys = vys[z_markers_idx]
        if len(fcs) > 1:
            fcs = fcs[z_markers_idx]
        if len(ecs) > 1:
            ecs = ecs[z_markers_idx]
        if len(sizes) > 1:
            sizes = sizes[z_markers_idx]
        if len(lws) > 1:
            lws = lws[z_markers_idx]
        vps = np.column_stack((vxs, vys))

        fcs = mcolors.to_rgba_array(fcs, self._alpha)
        ecs = mcolors.to_rgba_array(ecs, self._alpha)

        self.set_edgecolors(ecs)
        self.set_facecolors(fcs)
        self.set_sizes(sizes)
        self.set_linewidth(lws)

        PathCollection.set_offsets(self, vps)

        return np.min(vzs) if vzs.size else np.nan


def _update_scalarmappable(sm):
    """
    Update a 3D ScalarMappable.

    With ScalarMappable objects if the data, colormap, or norm are
    changed, we need to update the computed colors.  This is handled
    by the base class method update_scalarmappable.  This method works
    by, detecting if work needs to be done, and if so stashing it on
    the ``self._facecolors`` attribute.

    With 3D collections we internally sort the components so that
    things that should be "in front" are rendered later to simulate
    having a z-buffer (in addition to doing the projections).  This is
    handled in the ``do_3d_projection`` methods which are called from the
    draw method of the 3D Axes.  These methods:

    - do the projection from 3D -> 2D
    - internally sort based on depth
    - stash the results of the above in the 2D analogs of state
    - return the z-depth of the whole artist

    the last step is so that we can, at the Axes level, sort the children by
    depth.

    The base `draw` method of the 2D artists unconditionally calls
    update_scalarmappable and rely on the method's internal caching logic to
    lazily evaluate.

    These things together mean you can have the sequence of events:

    - we create the artist, do the color mapping and stash the results
      in a 3D specific state.
    - change something about the ScalarMappable that marks it as in
      need of an update (`ScalarMappable.changed` and friends).
    - We call do_3d_projection and shuffle the stashed colors into the
      2D version of face colors
    - the draw method calls the update_scalarmappable method which
      overwrites our shuffled colors
    - we get a render that is wrong
    - if we re-render (either with a second save or implicitly via
      tight_layout / constrained_layout / bbox_inches='tight' (ex via
      inline's defaults)) we again shuffle the 3D colors
    - because the CM is not marked as changed update_scalarmappable is
      a no-op and we get a correct looking render.

    This function is an internal helper to:

    - sort out if we need to do the color mapping at all (has data!)
    - sort out if update_scalarmappable is going to be a no-op
    - copy the data over from the 2D -> 3D version

    This must be called first thing in do_3d_projection to make sure that
    the correct colors get shuffled.

    Parameters
    ----------
    sm : ScalarMappable
        The ScalarMappable to update and stash the 3D data from

    """
    if sm._A is None:
        return
    copy_state = sm._update_dict['array']
    ret = sm.update_scalarmappable()
    if copy_state:
        if sm._is_filled:
            sm._facecolor3d = sm._facecolors
        elif sm._is_stroked:
            sm._edgecolor3d = sm._edgecolors


def patch_collection_2d_to_3d(col, zs=0, zdir='z', depthshade=True):
    """
    Convert a :class:`~matplotlib.collections.PatchCollection` into a
    :class:`Patch3DCollection` object
    (or a :class:`~matplotlib.collections.PathCollection` into a
    :class:`Path3DCollection` object).

    Parameters
    ----------
    za
        The location or locations to place the patches in the collection along
        the *zdir* axis. Default: 0.
    zdir
        The axis in which to place the patches. Default: "z".
    depthshade
        Whether to shade the patches to give a sense of depth. Default: *True*.

    """
    if isinstance(col, PathCollection):
        col.__class__ = Path3DCollection
    elif isinstance(col, PatchCollection):
        col.__class__ = Patch3DCollection
    col._depthshade = depthshade
    col.set_3d_properties(zs, zdir)


class Poly3DCollection(PolyCollection):
    """
    A collection of 3D polygons.

    .. note::
        **Filling of 3D polygons**

        There is no simple definition of the enclosed surface of a 3D polygon
        unless the polygon is planar.

        In practice, Matplotlib fills the 2D projection of the polygon. This
        gives a correct filling appearance only for planar polygons. For all
        other polygons, you'll find orientations in which the edges of the
        polygon intersect in the projection. This will lead to an incorrect
        visualization of the 3D area.

        If you need filled areas, it is recommended to create them via
        `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
        triangulation and thus generates consistent surfaces.
    """

    def __init__(self, verts, *args, zsort='average', **kwargs):
        """
        Parameters
        ----------
        verts : list of array-like Nx3
            Each element describes a polygon as a sequence of ``N_i`` points
            ``(x, y, z)``.
        zsort : {'average', 'min', 'max'}, default: 'average'
            The calculation method for the z-order.
            See `~.Poly3DCollection.set_zsort` for details.
        *args, **kwargs
            All other parameters are forwarded to `.PolyCollection`.

        Notes
        -----
        Note that this class does a bit of magic with the _facecolors
        and _edgecolors properties.
        """
        super().__init__(verts, *args, **kwargs)
        self.set_zsort(zsort)
        self._codes3d = None

    _zsort_functions = {
        'average': np.average,
        'min': np.min,
        'max': np.max,
    }

    def set_zsort(self, zsort):
        """
        Set the calculation method for the z-order.

        Parameters
        ----------
        zsort : {'average', 'min', 'max'}
            The function applied on the z-coordinates of the vertices in the
            viewer's coordinate system, to determine the z-order.
        """
        self._zsortfunc = self._zsort_functions[zsort]
        self._sort_zpos = None
        self.stale = True

    def get_vector(self, segments3d):
        """Optimize points for projection."""
        if len(segments3d):
            xs, ys, zs = np.row_stack(segments3d).T
        else:  # row_stack can't stack zero arrays.
            xs, ys, zs = [], [], []
        ones = np.ones(len(xs))
        self._vec = np.array([xs, ys, zs, ones])

        indices = [0, *np.cumsum([len(segment) for segment in segments3d])]
        self._segslices = [*map(slice, indices[:-1], indices[1:])]

    def set_verts(self, verts, closed=True):
        """Set 3D vertices."""
        self.get_vector(verts)
        # 2D verts will be updated at draw time
        PolyCollection.set_verts(self, [], False)
        self._closed = closed

    def set_verts_and_codes(self, verts, codes):
        """Set 3D vertices with path codes."""
        # set vertices with closed=False to prevent PolyCollection from
        # setting path codes
        self.set_verts(verts, closed=False)
        # and set our own codes instead.
        self._codes3d = codes

    def set_3d_properties(self):
        # Force the collection to initialize the face and edgecolors
        # just in case it is a scalarmappable with a colormap.
        self.update_scalarmappable()
        self._sort_zpos = None
        self.set_zsort('average')
        self._facecolor3d = PolyCollection.get_facecolor(self)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)
        self._alpha3d = PolyCollection.get_alpha(self)
        self.stale = True

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def do_3d_projection(self, renderer):
        """
        Perform the 3D projection for this object.
        """
        # see _update_scalarmappable docstring for why this must be here
        _update_scalarmappable(self)

        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, renderer.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]

        # This extra fuss is to re-order face / edge colors
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)

        # sort by depth (furthest drawn first)
        z_segments_2d = sorted(
            ((self._zsortfunc(zs), np.column_stack([xs, ys]), fc, ec, idx)
             for idx, ((xs, ys, zs), fc, ec)
             in enumerate(zip(xyzlist, cface, cedge))),
            key=lambda x: x[0], reverse=True)

        zzs, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = \
            zip(*z_segments_2d)

        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)

        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d

        # Return zorder value
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, renderer.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            # FIXME: Some results still don't look quite right.
            #        In particular, examine contourf3d_demo2.py
            #        with az = -54 and elev = -45.
            return np.min(tzs)
        else:
            return np.nan

    def set_facecolor(self, colors):
        PolyCollection.set_facecolor(self, colors)
        self._facecolor3d = PolyCollection.get_facecolor(self)

    def set_edgecolor(self, colors):
        PolyCollection.set_edgecolor(self, colors)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)

    def set_alpha(self, alpha):
        # docstring inherited
        artist.Artist.set_alpha(self, alpha)
        try:
            self._facecolor3d = mcolors.to_rgba_array(
                self._facecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        try:
            self._edgecolors = mcolors.to_rgba_array(
                    self._edgecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        self.stale = True

    def get_facecolor(self):
        return self._facecolors2d

    def get_edgecolor(self):
        return self._edgecolors2d


def poly_collection_2d_to_3d(col, zs=0, zdir='z'):
    """Convert a PolyCollection to a Poly3DCollection object."""
    segments_3d, codes = _paths_to_3d_segments_with_codes(
            col.get_paths(), zs, zdir)
    col.__class__ = Poly3DCollection
    col.set_verts_and_codes(segments_3d, codes)
    col.set_3d_properties()


def juggle_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that 2D xs, ys can be plotted in the plane
    orthogonal to zdir. zdir is normally x, y or z. However, if zdir
    starts with a '-' it is interpreted as a compensation for rotate_axes.
    """
    if zdir == 'x':
        return zs, xs, ys
    elif zdir == 'y':
        return xs, zs, ys
    elif zdir[0] == '-':
        return rotate_axes(xs, ys, zs, zdir)
    else:
        return xs, ys, zs


def rotate_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that the axes are rotated with zdir along
    the original z axis. Prepending the axis with a '-' does the
    inverse transform, so zdir can be x, -x, y, -y, z or -z
    """
    if zdir == 'x':
        return ys, zs, xs
    elif zdir == '-x':
        return zs, xs, ys

    elif zdir == 'y':
        return zs, xs, ys
    elif zdir == '-y':
        return ys, zs, xs

    else:
        return xs, ys, zs


def _get_colors(c, num):
    """Stretch the color argument to provide the required number *num*."""
    return np.broadcast_to(
        mcolors.to_rgba_array(c) if len(c) else [0, 0, 0, 0],
        (num, 4))


def _zalpha(colors, zs):
    """Modify the alphas of the color list according to depth."""
    # FIXME: This only works well if the points for *zs* are well-spaced
    #        in all three dimensions. Otherwise, at certain orientations,
    #        the min and max zs are very close together.
    #        Should really normalize against the viewing depth.
    if len(colors) == 0 or len(zs) == 0:
        return np.zeros((0, 4))
    norm = Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(mcolors.to_rgba_array(colors), (len(zs), 4))
    return np.column_stack([rgba[:, :3], rgba[:, 3] * sats])
