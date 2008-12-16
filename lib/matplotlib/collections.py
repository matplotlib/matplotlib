"""
Classes for the efficient drawing of large collections of objects that
share most properties, e.g. a large number of line segments or
polygons.

The classes are not meant to be as flexible as their single element
counterparts (e.g. you may not be able to select all line styles) but
they are meant to be fast for common use cases (e.g. a bunch of solid
line segemnts)
"""
import copy, math, warnings
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as _colors # avoid conflict with kwarg
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import matplotlib.artist as artist
import matplotlib.backend_bases as backend_bases
import matplotlib.path as mpath
import matplotlib.mlab as mlab

class Collection(artist.Artist, cm.ScalarMappable):
    """
    Base class for Collections.  Must be subclassed to be usable.

    All properties in a collection must be sequences or scalars;
    if scalars, they will be converted to sequences.  The
    property of the ith element of the collection is::

      prop[i % len(props)]

    Keyword arguments and default values:

        * *edgecolors*: None
        * *facecolors*: None
        * *linewidths*: None
        * *antialiaseds*: None
        * *offsets*: None
        * *transOffset*: transforms.IdentityTransform()
        * *norm*: None (optional for
          :class:`matplotlib.cm.ScalarMappable`)
        * *cmap*: None (optional for
          :class:`matplotlib.cm.ScalarMappable`)

    *offsets* and *transOffset* are used to translate the patch after
    rendering (default no offsets).

    If any of *edgecolors*, *facecolors*, *linewidths*, *antialiaseds*
    are None, they default to their :data:`matplotlib.rcParams` patch
    setting, in sequence form.

    The use of :class:`~matplotlib.cm.ScalarMappable` is optional.  If
    the :class:`~matplotlib.cm.ScalarMappable` matrix _A is not None
    (ie a call to set_array has been made), at draw time a call to
    scalar mappable will be made to set the face colors.
    """
    _offsets = np.array([], np.float_)
    _transOffset = transforms.IdentityTransform()
    _transforms = []

    zorder = 1
    def __init__(self,
                 edgecolors=None,
                 facecolors=None,
                 linewidths=None,
                 linestyles='solid',
                 antialiaseds = None,
                 offsets = None,
                 transOffset = None,
                 norm = None,  # optional for ScalarMappable
                 cmap = None,  # ditto
                 pickradius = 5.0,
                 urls = None,
                 **kwargs
                 ):
        """
        Create a Collection

        %(Collection)s
        """
        artist.Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        self.set_edgecolor(edgecolors)
        self.set_facecolor(facecolors)
        self.set_linewidth(linewidths)
        self.set_linestyle(linestyles)
        self.set_antialiased(antialiaseds)
        self.set_urls(urls)

        self._uniform_offsets = None
        self._offsets = np.array([], np.float_)
        if offsets is not None:
            offsets = np.asarray(offsets)
            if len(offsets.shape) == 1:
                offsets = offsets[np.newaxis,:]  # Make it Nx2.
            if transOffset is not None:
                self._offsets = offsets
                self._transOffset = transOffset
            else:
                self._uniform_offsets = offsets

        self._pickradius = pickradius
        self.update(kwargs)

    def _get_value(self, val):
        try: return (float(val), )
        except TypeError:
            if cbook.iterable(val) and len(val):
                try: float(val[0])
                except TypeError: pass # raise below
                else: return val

        raise TypeError('val must be a float or nonzero sequence of floats')

    def _get_bool(self, val):
        try: return (bool(val), )
        except TypeError:
            if cbook.iterable(val) and len(val):
                try: bool(val[0])
                except TypeError: pass # raise below
                else: return val

        raise TypeError('val must be a bool or nonzero sequence of them')


    def get_paths(self):
        raise NotImplementedError

    def get_transforms(self):
        return self._transforms

    def get_datalim(self, transData):
        transform = self.get_transform()
        transOffset = self._transOffset
        offsets = self._offsets
        paths = self.get_paths()
        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(p) for p in paths]
            transform = transform.get_affine()
        if not transOffset.is_affine:
            offsets = transOffset.transform_non_affine(offsets)
            transOffset = transOffset.get_affine()
        offsets = np.asarray(offsets, np.float_)

        result = mpath.get_path_collection_extents(
            transform.frozen(), paths, self.get_transforms(),
            offsets, transOffset.frozen())
        result = result.inverse_transformed(transData)
        return result

    def get_window_extent(self, renderer):
        bbox = self.get_datalim(transforms.IdentityTransform())
        #TODO:check to ensure that this does not fail for
        #cases other than scatter plot legend
        return bbox

    def _prepare_points(self):
        """Point prep for drawing and hit testing"""

        transform = self.get_transform()
        transOffset = self._transOffset
        offsets = self._offsets
        paths = self.get_paths()

        if self.have_units():
            paths = []
            for path in self.get_paths():
                vertices = path.vertices
                xs, ys = vertices[:, 0], vertices[:, 1]
                xs = self.convert_xunits(xs)
                ys = self.convert_yunits(ys)
                paths.append(mpath.Path(zip(xs, ys), path.codes))
            if len(self._offsets):
                xs = self.convert_xunits(self._offsets[:0])
                ys = self.convert_yunits(self._offsets[:1])
                offsets = zip(xs, ys)

        offsets = np.asarray(offsets, np.float_)

        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(path) for path in paths]
            transform = transform.get_affine()
        if not transOffset.is_affine:
            offsets = transOffset.transform_non_affine(offsets)
            transOffset = transOffset.get_affine()

        return transform, transOffset, offsets, paths

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group(self.__class__.__name__)

        self.update_scalarmappable()

        clippath, clippath_trans = self.get_transformed_clip_path_and_affine()
        if clippath_trans is not None:
            clippath_trans = clippath_trans.frozen()

        transform, transOffset, offsets, paths = self._prepare_points()

        renderer.draw_path_collection(
            transform.frozen(), self.clipbox, clippath, clippath_trans,
            paths, self.get_transforms(),
            offsets, transOffset,
            self.get_facecolor(), self.get_edgecolor(), self._linewidths,
            self._linestyles, self._antialiaseds, self._urls)
        renderer.close_group(self.__class__.__name__)

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns True | False, ``dict(ind=itemlist)``, where every
        item in itemlist contains the event.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        if not self.get_visible(): return False,{}

        transform, transOffset, offsets, paths = self._prepare_points()

        ind = mpath.point_in_path_collection(
            mouseevent.x, mouseevent.y, self._pickradius,
            transform.frozen(), paths, self.get_transforms(),
            offsets, transOffset, len(self._facecolors)>0)
        return len(ind)>0,dict(ind=ind)

    def set_pickradius(self,pickradius): self.pickradius = 5
    def get_pickradius(self): return self.pickradius

    def set_urls(self, urls):
	if urls is None:
            self._urls = [None,]
        else:
            self._urls = urls

    def get_urls(self): return self._urls

    def set_offsets(self, offsets):
        """
        Set the offsets for the collection.  *offsets* can be a scalar
        or a sequence.

        ACCEPTS: float or sequence of floats
        """
        offsets = np.asarray(offsets, np.float_)
        if len(offsets.shape) == 1:
            offsets = offsets[np.newaxis,:]  # Make it Nx2.
        #This decision is based on how they are initialized above
        if self._uniform_offsets is None:
            self._offsets = offsets
        else:
            self._uniform_offsets = offsets

    def get_offsets(self):
        """
        Return the offsets for the collection.
        """
        #This decision is based on how they are initialized above in __init__()
        if self._uniform_offsets is None:
            return self._offsets
        else:
            return self._uniform_offsets

    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.  *lw* can be a scalar
        or a sequence; if it is a sequence the patches will cycle
        through the sequence

        ACCEPTS: float or sequence of floats
        """
        if lw is None: lw = mpl.rcParams['patch.linewidth']
        self._linewidths = self._get_value(lw)

    def set_linewidths(self, lw):
        """alias for set_linewidth"""
        return self.set_linewidth(lw)

    def set_lw(self, lw):
        """alias for set_linewidth"""
        return self.set_linewidth(lw)

    def set_linestyle(self, ls):
        """
        Set the linestyle(s) for the collection.

        ACCEPTS: ['solid' | 'dashed', 'dashdot', 'dotted' |
        (offset, on-off-dash-seq) ]
        """
        try:
            dashd = backend_bases.GraphicsContextBase.dashd
            if cbook.is_string_like(ls):
                if ls in dashd:
                    dashes = [dashd[ls]]
                elif ls in cbook.ls_mapper:
                    dashes = [dashd[cbook.ls_mapper[ls]]]
                else:
                    raise ValueError()
            elif cbook.iterable(ls):
                try:
                    dashes = []
                    for x in ls:
                        if cbook.is_string_like(x):
                            if x in dashd:
                                dashes.append(dashd[x])
                            elif x in cbook.ls_mapper:
                                dashes.append(dashd[cbook.ls_mapper[x]])
                            else:
                                raise ValueError()
                        elif cbook.iterable(x) and len(x) == 2:
                            dashes.append(x)
                        else:
                            raise ValueError()
                except ValueError:
                    if len(ls)==2:
                        dashes = ls
                    else:
                        raise ValueError()
            else:
                raise ValueError()
        except ValueError:
            raise ValueError('Do not know how to convert %s to dashes'%ls)
        self._linestyles = dashes

    def set_linestyles(self, ls):
        """alias for set_linestyle"""
        return self.set_linestyle(ls)

    def set_dashes(self, ls):
        """alias for set_linestyle"""
        return self.set_linestyle(ls)

    def set_antialiased(self, aa):
        """
        Set the antialiasing state for rendering.

        ACCEPTS: Boolean or sequence of booleans
        """
        if aa is None:
            aa = mpl.rcParams['patch.antialiased']
        self._antialiaseds = self._get_bool(aa)

    def set_antialiaseds(self, aa):
        """alias for set_antialiased"""
        return self.set_antialiased(aa)

    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.

        ACCEPTS: matplotlib color arg or sequence of rgba tuples

        .. seealso::
            :meth:`set_facecolor`, :meth:`set_edgecolor`
        """
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_facecolor(self, c):
        """
        Set the facecolor(s) of the collection.  *c* can be a
        matplotlib color arg (all patches have same color), or a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        if c is None: c = mpl.rcParams['patch.facecolor']
        self._facecolors_original = c
        self._facecolors = _colors.colorConverter.to_rgba_array(c, self._alpha)

    def set_facecolors(self, c):
        """alias for set_facecolor"""
        return self.set_facecolor(c)

    def get_facecolor(self):
        return self._facecolors
    get_facecolors = get_facecolor

    def get_edgecolor(self):
        if self._edgecolors == 'face':
            return self.get_facecolors()
        else:
            return self._edgecolors
    get_edgecolors = get_edgecolor

    def set_edgecolor(self, c):
        """
        Set the edgecolor(s) of the collection. *c* can be a
        matplotlib color arg (all patches have same color), or a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence.

        If *c* is 'face', the edge color will always be the same as
        the face color.

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        if c == 'face':
            self._edgecolors = 'face'
            self._edgecolors_original = 'face'
        else:
            if c is None: c = mpl.rcParams['patch.edgecolor']
            self._edgecolors_original = c
            self._edgecolors = _colors.colorConverter.to_rgba_array(c, self._alpha)

    def set_edgecolors(self, c):
        """alias for set_edgecolor"""
        return self.set_edgecolor(c)

    def set_alpha(self, alpha):
        """
        Set the alpha tranparencies of the collection.  *alpha* must be
        a float.

        ACCEPTS: float
        """
        try: float(alpha)
        except TypeError: raise TypeError('alpha must be a float')
        else:
            artist.Artist.set_alpha(self, alpha)
            try:
                self._facecolors = _colors.colorConverter.to_rgba_array(
                    self._facecolors_original, self._alpha)
            except (AttributeError, TypeError, IndexError):
                pass
            try:
                if self._edgecolors_original != 'face':
                    self._edgecolors = _colors.colorConverter.to_rgba_array(
                        self._edgecolors_original, self._alpha)
            except (AttributeError, TypeError, IndexError):
                pass

    def get_linewidths(self):
        return self._linewidths
    get_linewidth = get_linewidths

    def get_linestyles(self):
        return self._linestyles
    get_dashes = get_linestyle = get_linestyles

    def update_scalarmappable(self):
        """
        If the scalar mappable array is not none, update colors
        from scalar data
        """
        if self._A is None: return
        if self._A.ndim > 1:
            raise ValueError('Collections can only map rank 1 arrays')
        if len(self._facecolors):
            self._facecolors = self.to_rgba(self._A, self._alpha)
        else:
            self._edgecolors = self.to_rgba(self._A, self._alpha)

    def update_from(self, other):
        'copy properties from other to self'

        artist.Artist.update_from(self, other)
        self._antialiaseds = other._antialiaseds
        self._edgecolors_original = other._edgecolors_original
        self._edgecolors = other._edgecolors
        self._facecolors_original = other._facecolors_original
        self._facecolors = other._facecolors
        self._linewidths = other._linewidths
        self._linestyles = other._linestyles
        self._pickradius = other._pickradius

# these are not available for the object inspector until after the
# class is built so we define an initial set here for the init
# function and they will be overridden after object defn
artist.kwdocd['Collection'] = """\
    Valid Collection keyword arguments:

        * *edgecolors*: None
        * *facecolors*: None
        * *linewidths*: None
        * *antialiaseds*: None
        * *offsets*: None
        * *transOffset*: transforms.IdentityTransform()
        * *norm*: None (optional for
          :class:`matplotlib.cm.ScalarMappable`)
        * *cmap*: None (optional for
          :class:`matplotlib.cm.ScalarMappable`)

    *offsets* and *transOffset* are used to translate the patch after
    rendering (default no offsets)

    If any of *edgecolors*, *facecolors*, *linewidths*, *antialiaseds*
    are None, they default to their :data:`matplotlib.rcParams` patch
    setting, in sequence form.
"""

class QuadMesh(Collection):
    """
    Class for the efficient drawing of a quadrilateral mesh.

    A quadrilateral mesh consists of a grid of vertices. The
    dimensions of this array are (*meshWidth* + 1, *meshHeight* +
    1). Each vertex in the mesh has a different set of "mesh
    coordinates" representing its position in the topology of the
    mesh. For any values (*m*, *n*) such that 0 <= *m* <= *meshWidth*
    and 0 <= *n* <= *meshHeight*, the vertices at mesh coordinates
    (*m*, *n*), (*m*, *n* + 1), (*m* + 1, *n* + 1), and (*m* + 1, *n*)
    form one of the quadrilaterals in the mesh. There are thus
    (*meshWidth* * *meshHeight*) quadrilaterals in the mesh.  The mesh
    need not be regular and the polygons need not be convex.

    A quadrilateral mesh is represented by a (2 x ((*meshWidth* + 1) *
    (*meshHeight* + 1))) numpy array *coordinates*, where each row is
    the *x* and *y* coordinates of one of the vertices.  To define the
    function that maps from a data point to its corresponding color,
    use the :meth:`set_cmap` method.  Each of these arrays is indexed in
    row-major order by the mesh coordinates of the vertex (or the mesh
    coordinates of the lower left vertex, in the case of the
    colors).

    For example, the first entry in *coordinates* is the
    coordinates of the vertex at mesh coordinates (0, 0), then the one
    at (0, 1), then at (0, 2) .. (0, meshWidth), (1, 0), (1, 1), and
    so on.
    """
    def __init__(self, meshWidth, meshHeight, coordinates, showedges, antialiased=True):
        Collection.__init__(self)
        self._meshWidth = meshWidth
        self._meshHeight = meshHeight
        self._coordinates = coordinates
        self._showedges = showedges
        self._antialiased = antialiased

        self._paths = None

        self._bbox = transforms.Bbox.unit()
        self._bbox.update_from_data_xy(coordinates.reshape(
                ((meshWidth + 1) * (meshHeight + 1), 2)))

        # By converting to floats now, we can avoid that on every draw.
        self._coordinates = self._coordinates.reshape((meshHeight + 1, meshWidth + 1, 2))
        self._coordinates = np.array(self._coordinates, np.float_)

    def get_paths(self, dataTrans=None):
        if self._paths is None:
            self._paths = self.convert_mesh_to_paths(
                self._meshWidth, self._meshHeight, self._coordinates)
        return self._paths

    #@staticmethod
    def convert_mesh_to_paths(meshWidth, meshHeight, coordinates):
        """
        Converts a given mesh into a sequence of
        :class:`matplotlib.path.Path` objects for easier rendering by
        backends that do not directly support quadmeshes.

        This function is primarily of use to backend implementers.
        """
        Path = mpath.Path

        if ma.isMaskedArray(coordinates):
            c = coordinates.data
        else:
            c = coordinates

        points = np.concatenate((
                    c[0:-1, 0:-1],
                    c[0:-1, 1:  ],
                    c[1:  , 1:  ],
                    c[1:  , 0:-1],
                    c[0:-1, 0:-1]
                    ), axis=2)
        points = points.reshape((meshWidth * meshHeight, 5, 2))
        return [Path(x) for x in points]
    convert_mesh_to_paths = staticmethod(convert_mesh_to_paths)

    def get_datalim(self, transData):
        return self._bbox

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group(self.__class__.__name__)
        transform = self.get_transform()
        transOffset = self._transOffset
        offsets = self._offsets

        if self.have_units():
            if len(self._offsets):
                xs = self.convert_xunits(self._offsets[:0])
                ys = self.convert_yunits(self._offsets[:1])
                offsets = zip(xs, ys)

        offsets = np.asarray(offsets, np.float_)

        if self.check_update('array'):
            self.update_scalarmappable()

        clippath, clippath_trans = self.get_transformed_clip_path_and_affine()
        if clippath_trans is not None:
            clippath_trans = clippath_trans.frozen()

        if not transform.is_affine:
            coordinates = self._coordinates.reshape(
                (self._coordinates.shape[0] *
                 self._coordinates.shape[1],
                 2))
            coordinates = transform.transform(coordinates)
            coordinates = coordinates.reshape(self._coordinates.shape)
            transform = transforms.IdentityTransform()
        else:
            coordinates = self._coordinates

        if not transOffset.is_affine:
            offsets = transOffset.transform_non_affine(offsets)
            transOffset = transOffset.get_affine()

        renderer.draw_quad_mesh(
            transform.frozen(), self.clipbox, clippath, clippath_trans,
            self._meshWidth, self._meshHeight, coordinates,
            offsets, transOffset, self.get_facecolor(), self._antialiased,
            self._showedges)
        renderer.close_group(self.__class__.__name__)

class PolyCollection(Collection):
    def __init__(self, verts, sizes = None, closed = True, **kwargs):
        """
        *verts* is a sequence of ( *verts0*, *verts1*, ...) where
        *verts_i* is a sequence of *xy* tuples of vertices, or an
        equivalent :mod:`numpy` array of shape (*nv*, 2).

        *sizes* is *None* (default) or a sequence of floats that
        scale the corresponding *verts_i*.  The scaling is applied
        before the Artist master transform; if the latter is an identity
        transform, then the overall scaling is such that if
        *verts_i* specify a unit square, then *sizes_i* is the area
        of that square in points^2.
        If len(*sizes*) < *nv*, the additional values will be
        taken cyclically from the array.

        *closed*, when *True*, will explicitly close the polygon.

        %(Collection)s
        """
        Collection.__init__(self,**kwargs)
        self._sizes = sizes
        self.set_verts(verts, closed)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def set_verts(self, verts, closed=True):
        '''This allows one to delay initialization of the vertices.'''
        if closed:
            self._paths = []
            for xy in verts:
                if np.ma.isMaskedArray(xy):
                    if len(xy) and (xy[0] != xy[-1]).any():
                        xy = np.ma.concatenate([xy, [xy[0]]])
                else:
                    xy = np.asarray(xy)
                    if len(xy) and (xy[0] != xy[-1]).any():
                        xy = np.concatenate([xy, [xy[0]]])
                self._paths.append(mpath.Path(xy))
        else:
            self._paths = [mpath.Path(xy) for xy in verts]

    def get_paths(self):
        return self._paths

    def draw(self, renderer):
        if self._sizes is not None:
            self._transforms = [
                transforms.Affine2D().scale(
                    (np.sqrt(x) * self.figure.dpi / 72.0))
                for x in self._sizes]
        return Collection.draw(self, renderer)


class BrokenBarHCollection(PolyCollection):
    """
    A collection of horizontal bars spanning *yrange* with a sequence of
    *xranges*.
    """
    def __init__(self, xranges, yrange, **kwargs):
        """
        *xranges*
            sequence of (*xmin*, *xwidth*)

        *yrange*
            *ymin*, *ywidth*

        %(Collection)s
        """
        ymin, ywidth = yrange
        ymax = ymin + ywidth
        verts = [ [(xmin, ymin), (xmin, ymax), (xmin+xwidth, ymax), (xmin+xwidth, ymin), (xmin, ymin)] for xmin, xwidth in xranges]
        PolyCollection.__init__(self, verts, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd


    @staticmethod
    def span_where(x, ymin, ymax, where, **kwargs):
        """
        Create a BrokenBarHCollection to plot horizontal bars from
        over the regions in *x* where *where* is True.  The bars range
        on the y-axis from *ymin* to *ymax*

        A :class:`BrokenBarHCollection` is returned.
        *kwargs* are passed on to the collection
        """
        xranges = []
        for ind0, ind1 in mlab.contiguous_regions(where):
            xslice = x[ind0:ind1]
            if not len(xslice):
                continue
            xranges.append((xslice[0], xslice[-1]-xslice[0]))

        collection = BrokenBarHCollection(xranges, [ymin, ymax-ymin], **kwargs)
        return collection

class RegularPolyCollection(Collection):
    """Draw a collection of regular polygons with *numsides*."""
    _path_generator = mpath.Path.unit_regular_polygon

    def __init__(self,
                 numsides,
                 rotation = 0 ,
                 sizes = (1,),
                 **kwargs):
        """
        *numsides*
            the number of sides of the polygon

        *rotation*
            the rotation of the polygon in radians

        *sizes*
            gives the area of the circle circumscribing the
            regular polygon in points^2

        %(Collection)s

        Example: see :file:`examples/dynamic_collection.py` for
        complete example::

            offsets = np.random.rand(20,2)
            facecolors = [cm.jet(x) for x in np.random.rand(20)]
            black = (0,0,0,1)

            collection = RegularPolyCollection(
                numsides=5, # a pentagon
                rotation=0, sizes=(50,),
                facecolors = facecolors,
                edgecolors = (black,),
                linewidths = (1,),
                offsets = offsets,
                transOffset = ax.transData,
                )
        """
        Collection.__init__(self,**kwargs)
        self._sizes = sizes
        self._numsides = numsides
        self._paths = [self._path_generator(numsides)]
        self._rotation = rotation
        self.set_transform(transforms.IdentityTransform())

    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def draw(self, renderer):
        self._transforms = [
            transforms.Affine2D().rotate(-self._rotation).scale(
                (np.sqrt(x) * self.figure.dpi / 72.0) / np.sqrt(np.pi))
            for x in self._sizes]
        return Collection.draw(self, renderer)

    def get_paths(self):
        return self._paths

    def get_numsides(self):
        return self._numsides

    def get_rotation(self):
        return self._rotation

    def get_sizes(self):
        return self._sizes


class StarPolygonCollection(RegularPolyCollection):
    """
    Draw a collection of regular stars with *numsides* points."""

    _path_generator = mpath.Path.unit_regular_star


class AsteriskPolygonCollection(RegularPolyCollection):
    """
    Draw a collection of regular asterisks with *numsides* points."""

    _path_generator = mpath.Path.unit_regular_asterisk


class LineCollection(Collection):
    """
    All parameters must be sequences or scalars; if scalars, they will
    be converted to sequences.  The property of the ith line
    segment is::

       prop[i % len(props)]

    i.e., the properties cycle if the ``len`` of props is less than the
    number of segments.
    """
    zorder = 2
    def __init__(self, segments,     # Can be None.
                 linewidths    = None,
                 colors       = None,
                 antialiaseds  = None,
                 linestyles = 'solid',
                 offsets = None,
                 transOffset = None,
                 norm = None,
                 cmap = None,
                 pickradius = 5,
                 **kwargs
                 ):
        """
        *segments*
            a sequence of (*line0*, *line1*, *line2*), where::

                linen = (x0, y0), (x1, y1), ... (xm, ym)

            or the equivalent numpy array with two columns. Each line
            can be a different length.

        *colors*
            must be a sequence of RGBA tuples (eg arbitrary color
            strings, etc, not allowed).

        *antialiaseds*
            must be a sequence of ones or zeros

        *linestyles* [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
            a string or dash tuple. The dash tuple is::

                (offset, onoffseq),

            where *onoffseq* is an even length tuple of on and off ink
            in points.

        If *linewidths*, *colors*, or *antialiaseds* is None, they
        default to their rcParams setting, in sequence form.

        If *offsets* and *transOffset* are not None, then
        *offsets* are transformed by *transOffset* and applied after
        the segments have been transformed to display coordinates.

        If *offsets* is not None but *transOffset* is None, then the
        *offsets* are added to the segments before any transformation.
        In this case, a single offset can be specified as::

            offsets=(xo,yo)

        and this value will be added cumulatively to each successive
        segment, so as to produce a set of successively offset curves.

        *norm*
            None (optional for :class:`matplotlib.cm.ScalarMappable`)
        *cmap*
            None (optional for :class:`matplotlib.cm.ScalarMappable`)

        *pickradius* is the tolerance for mouse clicks picking a line.
        The default is 5 pt.

        The use of :class:`~matplotlib.cm.ScalarMappable` is optional.
        If the :class:`~matplotlib.cm.ScalarMappable` matrix
        :attr:`~matplotlib.cm.ScalarMappable._A` is not None (ie a call to
        :meth:`~matplotlib.cm.ScalarMappable.set_array` has been made), at
        draw time a call to scalar mappable will be made to set the colors.
        """
        if colors is None: colors = mpl.rcParams['lines.color']
        if linewidths is None: linewidths = (mpl.rcParams['lines.linewidth'],)
        if antialiaseds is None: antialiaseds = (mpl.rcParams['lines.antialiased'],)
        self.set_linestyles(linestyles)

        colors = _colors.colorConverter.to_rgba_array(colors)

        Collection.__init__(
            self,
            edgecolors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            antialiaseds=antialiaseds,
            offsets=offsets,
            transOffset=transOffset,
            norm=norm,
            cmap=cmap,
            pickradius=pickradius,
            **kwargs)

        self.set_facecolors([])
        self.set_segments(segments)

    def get_paths(self):
        return self._paths

    def set_segments(self, segments):
        if segments is None: return
        _segments = []
        for seg in segments:
            if not np.ma.isMaskedArray(seg):
                seg = np.asarray(seg, np.float_)
            _segments.append(seg)
        if self._uniform_offsets is not None:
            _segments = self._add_offsets(_segments)
        self._paths = [mpath.Path(seg) for seg in _segments]

    set_verts = set_segments # for compatibility with PolyCollection

    def _add_offsets(self, segs):
        offsets = self._uniform_offsets
        Nsegs = len(segs)
        Noffs = offsets.shape[0]
        if Noffs == 1:
            for i in range(Nsegs):
                segs[i] = segs[i] + i * offsets
        else:
            for i in range(Nsegs):
                io = i%Noffs
                segs[i] = segs[i] + offsets[io:io+1]
        return segs

    def set_color(self, c):
        """
        Set the color(s) of the line collection.  *c* can be a
        matplotlib color arg (all patches have same color), or a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        self._edgecolors = _colors.colorConverter.to_rgba_array(c)

    def color(self, c):
        """
        Set the color(s) of the line collection.  *c* can be a
        matplotlib color arg (all patches have same color), or a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        warnings.warn('LineCollection.color deprecated; use set_color instead')
        return self.set_color(c)

    def get_color(self):
        return self._edgecolors
    get_colors = get_color  # for compatibility with old versions

class CircleCollection(Collection):
    """
    A collection of circles, drawn using splines.
    """
    def __init__(self, sizes, **kwargs):
        """
        *sizes*
            Gives the area of the circle in points^2

        %(Collection)s
        """
        Collection.__init__(self,**kwargs)
        self._sizes = sizes
        self.set_transform(transforms.IdentityTransform())
        self._paths = [mpath.Path.unit_circle()]
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def draw(self, renderer):
        # sizes is the area of the circle circumscribing the polygon
        # in points^2
        self._transforms = [
            transforms.Affine2D().scale(
                (np.sqrt(x) * self.figure.dpi / 72.0) / np.sqrt(np.pi))
            for x in self._sizes]
        return Collection.draw(self, renderer)

    def get_paths(self):
        return self._paths

class EllipseCollection(Collection):
    """
    A collection of ellipses, drawn using splines.
    """
    def __init__(self, widths, heights, angles, units='points', **kwargs):
        """
        *widths*: sequence
            half-lengths of first axes (e.g., semi-major axis lengths)

        *heights*: sequence
            half-lengths of second axes

        *angles*: sequence
            angles of first axes, degrees CCW from the X-axis

        *units*: ['points' | 'inches' | 'dots' | 'width' | 'height' | 'x' | 'y']
            units in which majors and minors are given; 'width' and 'height'
            refer to the dimensions of the axes, while 'x' and 'y'
            refer to the *offsets* data units.

        Additional kwargs inherited from the base :class:`Collection`:

        %(Collection)s
        """
        Collection.__init__(self,**kwargs)
        self._widths = np.asarray(widths).ravel()
        self._heights = np.asarray(heights).ravel()
        self._angles = np.asarray(angles).ravel() *(np.pi/180.0)
        self._units = units
        self.set_transform(transforms.IdentityTransform())
        self._transforms = []
        self._paths = [mpath.Path.unit_circle()]
        self._initialized = False


    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _init(self):
        def on_dpi_change(fig):
            self._transforms = []
        self.figure.callbacks.connect('dpi_changed', on_dpi_change)
        self._initialized = True

    def set_transforms(self):
        if not self._initialized:
            self._init()
        self._transforms = []
        ax = self.axes
        fig = self.figure
        if self._units in ('x', 'y'):
            if self._units == 'x':
                dx0 = ax.viewLim.width
                dx1 = ax.bbox.width
            else:
                dx0 = ax.viewLim.height
                dx1 = ax.bbox.height
            sc = dx1/dx0
        else:
            if self._units == 'inches':
                sc = fig.dpi
            elif self._units == 'points':
                sc = fig.dpi / 72.0
            elif self._units == 'width':
                sc = ax.bbox.width
            elif self._units == 'height':
                sc = ax.bbox.height
            elif self._units == 'dots':
                sc = 1.0
            else:
                raise ValueError('unrecognized units: %s' % self._units)

        _affine = transforms.Affine2D
        for x, y, a in zip(self._widths, self._heights, self._angles):
            trans = _affine().scale(x * sc, y * sc).rotate(a)
            self._transforms.append(trans)

    def draw(self, renderer):
        if True: ###not self._transforms:
            self.set_transforms()
        return Collection.draw(self, renderer)

    def get_paths(self):
        return self._paths

class PatchCollection(Collection):
    """
    A generic collection of patches.

    This makes it easier to assign a color map to a heterogeneous
    collection of patches.

    This also may improve plotting speed, since PatchCollection will
    draw faster than a large number of patches.
    """

    def __init__(self, patches, match_original=False, **kwargs):
        """
        *patches*
            a sequence of Patch objects.  This list may include
            a heterogeneous assortment of different patch types.

        *match_original*
            If True, use the colors and linewidths of the original
            patches.  If False, new colors may be assigned by
            providing the standard collection arguments, facecolor,
            edgecolor, linewidths, norm or cmap.

        If any of *edgecolors*, *facecolors*, *linewidths*,
        *antialiaseds* are None, they default to their
        :data:`matplotlib.rcParams` patch setting, in sequence form.

        The use of :class:`~matplotlib.cm.ScalarMappable` is optional.
        If the :class:`~matplotlib.cm.ScalarMappable` matrix _A is not
        None (ie a call to set_array has been made), at draw time a
        call to scalar mappable will be made to set the face colors.
        """

        if match_original:
            def determine_facecolor(patch):
                if patch.fill:
                    return patch.get_facecolor()
                return [0, 0, 0, 0]

            facecolors   = [determine_facecolor(p) for p in patches]
            edgecolors   = [p.get_edgecolor() for p in patches]
            linewidths   = [p.get_linewidths() for p in patches]
            antialiaseds = [p.get_antialiased() for p in patches]

            Collection.__init__(
                self,
                edgecolors=edgecolors,
                facecolors=facecolors,
                linewidths=linewidths,
                linestyles='solid',
                antialiaseds = antialiaseds)
        else:
            Collection.__init__(self, **kwargs)

        paths        = [p.get_transform().transform_path(p.get_path())
                        for p in patches]

        self._paths = paths

    def get_paths(self):
        return self._paths


artist.kwdocd['Collection'] = patchstr = artist.kwdoc(Collection)
for k in ('QuadMesh', 'PolyCollection', 'BrokenBarHCollection', 'RegularPolyCollection',
          'StarPolygonCollection', 'PatchCollection', 'CircleCollection'):
    artist.kwdocd[k] = patchstr
artist.kwdocd['LineCollection'] = artist.kwdoc(LineCollection)
