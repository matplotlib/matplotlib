"""
Classes for the efficient drawing of large collections of objects that
share most properties, eg a large number of line segments or polygons

The classes are not meant to be as flexible as their single element
counterparts (eg you may not be able to select all line styles) but
they are meant to be fast for common use cases (eg a bunch of solid
line segemnts)
"""
import math, warnings
import numpy as npy
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as _colors # avoid conflict with kwarg
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import matplotlib.artist as artist
import matplotlib.backend_bases as backend_bases
import matplotlib.nxutils as nxutils
import matplotlib.path as path

# MGDTODO: Move this stuff
from matplotlib.backends._backend_agg import get_path_collection_extents, \
    point_in_path_collection

class Collection(artist.Artist, cm.ScalarMappable):
    """
    Base class for Collections.  Must be subclassed to be usable.

    All properties in a collection must be sequences or scalars;
    if scalars, they will be converted to sequences.  The
    property of the ith element of the collection is the

      prop[i % len(props)].

    kwargs are:

          edgecolors=None,
          facecolors=None,
          linewidths=None,
          antialiaseds = None,
          offsets = None,
          transOffset = transforms.identity_transform(),
          norm = None,  # optional for cm.ScalarMappable
          cmap = None,  # ditto

    offsets and transOffset are used to translate the patch after
    rendering (default no offsets)

    If any of edgecolors, facecolors, linewidths, antialiaseds are
    None, they default to their patch.* rc params setting, in sequence
    form.

    The use of ScalarMappable is optional.  If the ScalarMappable
    matrix _A is not None (ie a call to set_array has been made), at
    draw time a call to scalar mappable will be made to set the face
    colors.
    """
    _offsets = npy.zeros((1, 2))
    _transOffset = transforms.IdentityTransform()

    _transforms = [None]
    
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
                 **kwargs
                 ):
        """
        Create a PatchCollection

        %(PatchCollection)s
        """
        artist.Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        if facecolors is None: facecolors = mpl.rcParams['patch.facecolor']
        if edgecolors is None: edgecolors = mpl.rcParams['patch.edgecolor']
        if linewidths is None: linewidths = (mpl.rcParams['patch.linewidth'],)
        if antialiaseds is None: antialiaseds = (mpl.rcParams['patch.antialiased'],)
        self.set_linestyles(linestyles)
        
        self._facecolors  = _colors.colorConverter.to_rgba_list(facecolors)
        if edgecolors == 'None':
            self._edgecolors = self._facecolors
            linewidths = (0,)
        else:
            self._edgecolors = _colors.colorConverter.to_rgba_list(edgecolors)
        self._linewidths  = self._get_value(linewidths)
        self._antialiaseds = self._get_value(antialiaseds)

        self._uniform_offsets = None
        self._offsets = npy.zeros((1, 2))
        if offsets is not None:
#             if len(offsets.shape) == 1:
#                 offsets = offsets[npy.newaxis,:]  # Make it Nx2.
            if transOffset is not None:
                Affine2D = transforms.Affine2D
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
        
    def get_paths(self):
        raise NotImplementedError

    def get_transforms(self):
        return self._transforms
        
    def get_datalim(self, transData):
        result = transforms.Bbox.from_lbrt(*get_path_collection_extents(
                self.get_transform().frozen(),
                self.get_paths(),
                self.get_transforms(),
                self._offsets,
                self._transOffset.frozen()))
        result = result.transformed(transData.inverted())
        return result

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group(self.__class__.__name__)
        transform = self.get_transform()
        transOffset = self._transOffset
        offsets = self._offsets
        paths = self.get_paths()

        # MGDTODO: Test me
        if self.have_units():
            paths = []
            for path in self._paths:
                vertices = path.vertices
                xs, ys = zip(*segment)
                xs = self.convert_xunits(xs)
                ys = self.convert_yunits(ys)
                paths.append(path.Path(zip(xs, ys), path.codes))
            if self._offsets is not None:
                xs = self.convert_xunits(self._offsets[:0])
                ys = self.convert_yunits(self._offsets[:1])
                offsets = zip(xs, ys)
                
        self.update_scalarmappable()

        #print 'calling renderer draw line collection'
        clippath, clippath_trans = self.get_transformed_clip_path_and_affine()

        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(path) for path in paths]
            transform = transform.get_affine()
        
        renderer.draw_path_collection(
            transform, self.clipbox, clippath, clippath_trans,
            paths, self.get_transforms(),
            npy.asarray(offsets, npy.float_), transOffset, 
            self._facecolors, self._edgecolors, self._linewidths,
            self._linestyles, self._antialiaseds)
        renderer.close_group(self.__class__.__name__)

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns T/F, dict(ind=itemlist), where every item in itemlist contains the event.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        ind = point_in_path_collection(
            mouseevent.x, mouseevent.y, self._pickradius,
            self.get_transform(), self._paths, self._transforms, self._offsets,
            self._offsetTrans, self._facecolors)
        return len(ind)>0,dict(ind=ind)

    # MGDTODO: Update
    def get_transformed_patches(self):
        """
        get a sequence of the polygons in the collection in display (transformed) space

        The ith element in the returned sequence is a list of x,y
        vertices defining the ith polygon
        """

        verts = self._verts
        offsets = self._offsets
        usingOffsets = offsets is not None
        transform = self.get_transform()
        transOffset = self.get_transoffset()
        Noffsets = 0
        Nverts = len(verts)
        if usingOffsets:
            Noffsets = len(offsets)

        N = max(Noffsets, Nverts)

        data = []
        #print 'verts N=%d, Nverts=%d'%(N, Nverts), verts
        #print 'offsets; Noffsets=%d'%Noffsets
        for i in xrange(N):
            #print 'i%%Nverts=%d'%(i%Nverts)
            polyverts = verts[i % Nverts]
            if npy.any(npy.isnan(polyverts)):
                continue
            #print 'thisvert', i, polyverts
            tverts = transform.seq_xy_tups(polyverts)
            if usingOffsets:
                #print 'using offsets'
                xo,yo = transOffset.xy_tup(offsets[i % Noffsets])
                tverts = [(x+xo,y+yo) for x,y in tverts]

            data.append(tverts)
        return data

    def set_pickradius(self,pickradius): self.pickradius = 5
    def get_pickradius(self): return self.pickradius
    
    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.  lw can be a scalar or a
        sequence; if it is a sequence the patches will cycle through the
        sequence

        ACCEPTS: float or sequence of floats
        """
        self._linewidths = self._get_value(lw)
    def set_linewidths(self, lw):
        self.set_linewidth(lw)

    def set_linestyles(self, ls):
        """
        Set the linestyles(s) for the collection.
        ACCEPTS: ['solid' | 'dashed', 'dashdot', 'dotted' |  (offset, on-off-dash-seq) ]
        """
        try:
            if cbook.is_string_like(ls):
                dashes = [backend_bases.GraphicsContextBase.dashd[ls]]
            elif cbook.iterable(ls):
                try:
                    dashes = []
                    for x in ls:
                        if cbook.is_string_like(x):
                            dashes.append(backend_bases.GraphicsContextBase.dashd[ls])
                        elif cbook.iterator(x) and len(x) == 2:
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
        
    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.
        See set_facecolor and set_edgecolor.

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_facecolor(self, c):
        """
        Set the facecolor(s) of the collection.  c can be a matplotlib
        color arg (all patches have same color), or a a sequence or
        rgba tuples; if it is a sequence the patches will cycle
        through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        self._facecolors = _colors.colorConverter.to_rgba_list(c)
    def set_facecolors(self, c):
        self.set_facecolor(c)

    def set_edgecolor(self, c):
        """
        Set the edgecolor(s) of the collection. c can be a matplotlib color
        arg (all patches have same color), or a a sequence or rgba tuples; if
        it is a sequence the patches will cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        if c == 'None':
            self._linewidths = (0.0,)
        else:
            self._edgecolors = _colors.colorConverter.to_rgba_list(c)
    def set_edgecolors(self, c):
        self.set_edgecolor(c)

    def set_alpha(self, alpha):
        """
        Set the alpha tranpancies of the collection.  Alpha must be
        a float.

        ACCEPTS: float
        """
        try: float(alpha)
        except TypeError: raise TypeError('alpha must be a float')
        else:
            artist.Artist.set_alpha(self, alpha)
            self._facecolors = [(r,g,b,alpha) for r,g,b,a in self._facecolors]
            if cbook.is_string_like(self._edgecolors) and self._edgecolors != 'None':
                self._edgecolors = [(r,g,b,alpha) for r,g,b,a in self._edgecolors]

    def get_linewidth(self):
        return self._linewidths
    
    def get_linestyle(self):
        return self._linestyles

    def get_dashes(self):
        return self._linestyles
                
    def update_scalarmappable(self):
        """
        If the scalar mappable array is not none, update facecolors
        from scalar data
        """
        #print 'update_scalarmappable: self._A', self._A
        if self._A is None: return
        if len(self._A.shape)>1:
            raise ValueError('PatchCollections can only map rank 1 arrays')
        self._facecolors = self.to_rgba(self._A, self._alpha)
        #print self._facecolors


# these are not available for the object inspector until after the
# class is built so we define an initial set here for the init
# function and they will be overridden after object defn
artist.kwdocd['Collection'] = """\
    Valid Collection kwargs are:

      edgecolors=None,
      facecolors=None,
      linewidths=None,
      antialiaseds = None,
      offsets = None,
      transOffset = transforms.identity_transform(),
      norm = None,  # optional for cm.ScalarMappable
      cmap = None,  # ditto

    offsets and transOffset are used to translate the patch after
    rendering (default no offsets)

    If any of edgecolors, facecolors, linewidths, antialiaseds are
    None, they default to their patch.* rc params setting, in sequence
    form.
"""
        
class QuadMesh(Collection):
    """
    Class for the efficient drawing of a quadrilateral mesh.
    A quadrilateral mesh consists of a grid of vertices. The dimensions
    of this array are (meshWidth+1, meshHeight+1). Each vertex in
    the mesh has a different set of "mesh coordinates" representing
    its position in the topology of the mesh. For any values (m, n)
    such that 0 <= m <= meshWidth and 0 <= n <= meshHeight, the
    vertices at mesh coordinates (m, n), (m, n+1), (m+1, n+1), and
    (m+1, n) form one of the quadrilaterals in the mesh. There are
    thus (meshWidth * meshHeight) quadrilaterals in the mesh.
    The mesh need not be regular and the polygons need not be convex.
    A quadrilateral mesh is represented by a
    (2 x ((meshWidth + 1) * (meshHeight + 1))) Numeric array
    'coordinates' where each row is the X and Y coordinates of one
    of the vertices.
    To define the function that maps from a data point to
    its corresponding color, use the set_cmap() function.
    Each of these arrays is indexed in row-major order by the
    mesh coordinates of the vertex (or the mesh coordinates of
    the lower left vertex, in the case of the colors). For example,
    the first entry in coordinates is the coordinates of the vertex
    at mesh coordinates (0, 0), then the one at (0, 1), then at
    (0, 2) .. (0, meshWidth), (1, 0), (1, 1), and so on.
    """
    def __init__(self, meshWidth, meshHeight, coordinates, showedges):
        Collection.__init__(self)
        self._meshWidth = meshWidth
        self._meshHeight = meshHeight
        self._coordinates = coordinates
        self._showedges = showedges

    def get_verts(self, dataTrans=None):
        return self._coordinates;

    def draw(self, renderer):
        # does not call update_scalarmappable, need to update it
        # when creating/changing              ****** Why not?  speed?
        if not self.get_visible(): return
        transform = self.get_transform()
        transoffset = self._transOffset
        transform.freeze()
        transoffset.freeze()
        #print 'QuadMesh draw'
        self.update_scalarmappable()  #######################

        renderer.draw_quad_mesh( self._meshWidth, self._meshHeight,
            self._facecolors, self._coordinates[:,0],
            self._coordinates[:, 1], self.clipbox, transform,
            self._offsets, transoffset, self._showedges)
        transform.thaw()
        transoffset.thaw()

class PolyCollection(Collection):
    def __init__(self, verts, **kwargs):
        """
        verts is a sequence of ( verts0, verts1, ...) where verts_i is
        a sequence of xy tuples of vertices, or an equivalent
        numpy array of shape (nv,2).

        %(Collection)s
        """
        Collection.__init__(self,**kwargs)
        self.set_verts(verts)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def set_verts(self, verts):
        '''This allows one to delay initialization of the vertices.'''
        self._paths = [path.Path(v, closed=True) for v in verts]

    def get_paths(self):
        return self._paths
        
class BrokenBarHCollection(PolyCollection):
    """
    A colleciton of horizontal bars spanning yrange with a sequence of
    xranges
    """
    def __init__(self, xranges, yrange, **kwargs):
        """
        xranges : sequence of (xmin, xwidth)
        yrange  : ymin, ywidth

        %(Collection)s
        """
        ymin, ywidth = yrange
        ymax = ymin + ywidth
        verts = [ [(xmin, ymin), (xmin, ymax), (xmin+xwidth, ymax), (xmin+xwidth, ymin)] for xmin, xwidth in xranges]
        PolyCollection.__init__(self, verts, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class RegularPolyCollection(Collection):
    def __init__(self,
                 dpi,
                 numsides,
                 rotation = 0 ,
                 sizes = (1,),
                 **kwargs):
        """
        Draw a regular polygon with numsides.

        * dpi is the figure dpi instance, and is required to do the
          area scaling.

        * numsides: the number of sides of the polygon

        * sizes gives the area of the circle circumscribing the
          regular polygon in points^2

        * rotation is the rotation of the polygon in radians

        %(Collection)s

        Example: see examples/dynamic_collection.py for complete example

        offsets = npy.random.rand(20,2)
        facecolors = [cm.jet(x) for x in npy.random.rand(20)]
        black = (0,0,0,1)

        collection = RegularPolyCollection(
            fig.dpi,
            numsides=5, # a pentagon
            rotation=0,
            sizes=(50,),
            facecolors = facecolors,
            edgecolors = (black,),
            linewidths = (1,),
            offsets = offsets,
            transOffset = ax.transData,
            )


        """
        Collection.__init__(self,**kwargs)
        self._sizes = sizes
        self._dpi = dpi
        self._paths = [path.Path.unit_regular_polygon(numsides)]
        # sizes is the area of the circle circumscribing the polygon
        # in points^2
        self._transforms = [
            transforms.Affine2D().rotate(rotation).scale(
                (math.sqrt(x) * self._dpi / 72.0) * (1.0 / math.sqrt(math.pi)))
            for x in sizes]
        self.set_transform(transforms.IdentityTransform())
        
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_paths(self):
        return self._paths
    
    # MGDTODO: Update
    def get_transformed_patches(self):
        # Shouldn't need all these calls to asarray;
        # the variables should be converted when stored.
        # Similar speedups with numpy should be attainable
        # in many other places.
        verts = npy.asarray(self._verts)
        offsets = npy.asarray(self._offsets)
        Npoly = len(offsets)
        scales = npy.sqrt(npy.asarray(self._sizes)*self._dpi.get()/72.0)
        Nscales = len(scales)
        if Nscales >1:
            scales = npy.resize(scales, (Npoly, 1, 1))
        transOffset = self.get_transoffset()
        xyo = transOffset.numerix_xy(offsets)
        polys = scales * verts + xyo[:, npy.newaxis, :]
        return polys


class StarPolygonCollection(RegularPolyCollection):
    def __init__(self,
                 dpi,
                 numsides,
                 rotation = 0 ,
                 sizes = (1,),
                 **kwargs):
        """
        Draw a regular star like Polygone with numsides.

        * dpi is the figure dpi instance, and is required to do the
          area scaling.

        * numsides: the number of sides of the polygon

        * sizes gives the area of the circle circumscribing the
          regular polygon in points^2

        * rotation is the rotation of the polygon in radians

        %(Collection)s
        """

        RegularPolyCollection.__init__(self, dpi, numsides, rotation, sizes, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _update_verts(self):
        scale = 1.0/math.sqrt(math.pi)
        ns2 = self.numsides*2
        r = scale*npy.ones(ns2)
        r[1::2] *= 0.5
        theta  = (math.pi/self.numsides)*npy.arange(ns2) + self.rotation
        self._verts = zip( r*npy.sin(theta), r*npy.cos(theta) )

class AsteriskPolygonCollection(RegularPolyCollection):
    def __init__(self,
                 dpi,
                 numsides,
                 rotation = 0 ,
                 sizes = (1,),
                 **kwargs):
        """
        Draw a regular asterisk Polygone with numsides spikes.

        * dpi is the figure dpi instance, and is required to do the
          area scaling.

        * numsides: the number of spikes of the polygon

        * sizes gives the area of the circle circumscribing the
          regular polygon in points^2

        * rotation is the rotation of the polygon in radians

        %(Collection)s
        """

        RegularPolyCollection.__init__(self, dpi, numsides, rotation, sizes, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _update_verts(self):
        scale = 1.0/math.sqrt(math.pi)
        r = scale*npy.ones(self.numsides*2)
        r[1::2] = 0
        theta  = (math.pi/self.numsides)*npy.arange(2*self.numsides) + self.rotation
        self._verts = zip( r*npy.sin(theta), r*npy.cos(theta) )

class LineCollection(Collection, cm.ScalarMappable):
    """
    All parameters must be sequences or scalars; if scalars, they will
    be converted to sequences.  The property of the ith line
    segment is the prop[i % len(props)], ie the properties cycle if
    the len of props is less than the number of sements
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
        segments is a sequence of ( line0, line1, line2), where
        linen = (x0, y0), (x1, y1), ... (xm, ym), or the
        equivalent numpy array with two columns.
        Each line can be a different length.

        colors must be a tuple of RGBA tuples (eg arbitrary color
        strings, etc, not allowed).

        antialiaseds must be a sequence of ones or zeros

        linestyles is a string or dash tuple. Legal string values are
          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
          where onoffseq is an even length tuple of on and off ink in points.

        If linewidths, colors_, or antialiaseds is None, they default to
        their rc params setting, in sequence form.

        If offsets and transOffset are not None, then
        offsets are transformed by transOffset and applied after
        the segments have been transformed to display coordinates.

        If offsets is not None but transOffset is None, then the
        offsets are added to the segments before any transformation.
        In this case, a single offset can be specified as offsets=(xo,yo),
        and this value will be
        added cumulatively to each successive segment, so as
        to produce a set of successively offset curves.

        norm = None,  # optional for ScalarMappable
        cmap = None,  # ditto

        pickradius is the tolerance for mouse clicks picking a line.  The
        default is 5 pt.

        The use of ScalarMappable is optional.  If the ScalarMappable
        matrix _A is not None (ie a call to set_array has been made), at
        draw time a call to scalar mappable will be made to set the colors.
        """
        if colors is None: colors = mpl.rcParams['lines.color']
        if linewidths is None: linewidths = (mpl.rcParams['lines.linewidth'],)
        if antialiaseds is None: antialiaseds = (mpl.rcParams['lines.antialiased'],)
        self.set_linestyles(linestyles)

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

        self._facecolors = [None]
        self.set_segments(segments)

    def get_paths(self):
        return self._paths
        
    def set_segments(self, segments):
        if segments is None: return
        segments = [npy.asarray(seg, npy.float_) for seg in segments]
        if self._uniform_offsets is not None:
            segments = self._add_offsets(segments)
        self._paths = [path.Path(seg, closed=False) for seg in segments]
        
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
        Set the color(s) of the line collection.  c can be a
        matplotlib color arg (all patches have same color), or a a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        self._edgecolors = _colors.colorConverter.to_rgba_list(c)

    def color(self, c):
        """
        Set the color(s) of the line collection.  c can be a
        matplotlib color arg (all patches have same color), or a a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        warnings.warn('LineCollection.color deprecated; use set_color instead')
        return self.set_color(c)

    def get_color(self):
        return self._edgecolors
    get_colors = get_color  # for compatibility with old versions


artist.kwdocd['Collection'] = patchstr = artist.kwdoc(Collection)
for k in ('QuadMesh', 'PolyCollection', 'BrokenBarHCollection', 'RegularPolyCollection',
          'StarPolygonCollection'):
    artist.kwdocd[k] = patchstr
artist.kwdocd['LineCollection'] = artist.kwdoc(LineCollection)

