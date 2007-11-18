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


class Collection(artist.Artist):
    """
    All properties in a collection must be sequences or scalars;
    if scalars, they will be converted to sequences.  The
    property of the ith element of the collection is the

      prop[i % len(props)].

    """

    def __init__(self):
        artist.Artist.__init__(self)


    def get_verts(self):
        'return seq of (x,y) in collection'
        raise NotImplementedError('Derived must override')

    def _get_value(self, val):
        try: return (float(val), )
        except TypeError:
            if cbook.iterable(val) and len(val):
                try: float(val[0])
                except TypeError: pass # raise below
                else: return val

        raise TypeError('val must be a float or nonzero sequence of floats')


# these are not available for the object inspector until after the
# class is built so we define an initial set here for the init
# function and they will be overridden after object defn
artist.kwdocd['PatchCollection'] = """\
    Valid PatchCollection kwargs are:

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

class PatchCollection(Collection, cm.ScalarMappable):
    """
    Base class for filled regions such as PolyCollection etc.
    It must be subclassed to be usable.

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
    zorder = 1
    def __init__(self,
                 edgecolors=None,
                 facecolors=None,
                 linewidths=None,
                 antialiaseds = None,
                 offsets = None,
                 transOffset = None,
                 norm = None,  # optional for ScalarMappable
                 cmap = None,  # ditto
                 ):
        """
        Create a PatchCollection

        %(PatchCollection)s
        """
        Collection.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        if facecolors is None: facecolors = mpl.rcParams['patch.facecolor']
        if edgecolors is None: edgecolors = mpl.rcParams['patch.edgecolor']
        if linewidths is None: linewidths = (mpl.rcParams['patch.linewidth'],)
        if antialiaseds is None: antialiaseds = (mpl.rcParams['patch.antialiased'],)

        self._facecolors  = _colors.colorConverter.to_rgba_list(facecolors)
        if edgecolors == 'None':
            self._edgecolors = self._facecolors
            linewidths = (0,)
        else:
            self._edgecolors = _colors.colorConverter.to_rgba_list(edgecolors)
        self._linewidths  = self._get_value(linewidths)
        self._antialiaseds = self._get_value(antialiaseds)
        #self._offsets = offsets
        self._offsets = offsets
        self._transOffset = transOffset
        self._verts = []

    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns T/F, dict(ind=itemlist), where every item in itemlist contains the event.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        # TODO: Consider doing the test in data coordinates
        # Patch transforms the mouse into data coordinates and does the
        # test for membership there.  This is more efficient though it
        # may not match the visual appearance of the polygon on the
        # screen.  Regardless, patch and patch collection should use
        # the same algorithm.  Here's the code in patch:
        #
        #    x, y = self.get_transform().inverse_xy_tup((mouseevent.x, mouseevent.y))
        #    xyverts = self.get_verts()
        #    inside = nxutils.pnpoly(x, y, xyverts)
        #
        ind = []
        x, y = mouseevent.x, mouseevent.y
        for i, thispoly in enumerate(self.get_transformed_patches()):
            inside = nxutils.pnpoly(x, y, thispoly)
            if inside: ind.append(i)
        return len(ind)>0,dict(ind=ind)


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

    def get_transoffset(self):
        if self._transOffset is None:
            self._transOffset = transforms.identity_transform()
        return self._transOffset


    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.  lw can be a scalar or a
        sequence; if it is a sequence the patches will cycle through the
        sequence

        ACCEPTS: float or sequence of floats
        """
        self._linewidths = self._get_value(lw)
    set_linewidths = set_lw = set_linewidth

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
        self._facecolors = _colors.colorConverter.to_rgba_list(c, self._alpha)

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

class QuadMesh(PatchCollection):
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
    (2 x ((meshWidth + 1) * (meshHeight + 1))) numpy array
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
        PatchCollection.__init__(self)
        self._meshWidth = meshWidth
        self._meshHeight = meshHeight
        self._coordinates = coordinates
        self._showedges = showedges

    def get_verts(self, dataTrans=None):
        return self._coordinates;

    def draw(self, renderer):
        if not self.get_visible(): return
        transform = self.get_transform()
        transoffset = self.get_transoffset()
        transform.freeze()
        transoffset.freeze()
        #print 'QuadMesh draw'
        if self.check_update('array'):
            self.update_scalarmappable()

        renderer.draw_quad_mesh( self._meshWidth, self._meshHeight,
            self._facecolors, self._coordinates[:,0],
            self._coordinates[:, 1], self.clipbox, transform,
            self._offsets, transoffset, self._showedges)
        transform.thaw()
        transoffset.thaw()

class PolyCollection(PatchCollection):
    def __init__(self, verts, **kwargs):
        """
        verts is a sequence of ( verts0, verts1, ...) where verts_i is
        a sequence of xy tuples of vertices, or an equivalent
        numpy array of shape (nv,2).

        %(PatchCollection)s
        """
        PatchCollection.__init__(self,**kwargs)
        self._verts = verts
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def set_verts(self, verts):
        '''This allows one to delay initialization of the vertices.'''
        self._verts = verts

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group('polycollection')
        transform = self.get_transform()
        transoffset = self.get_transoffset()


        transform.freeze()
        transoffset.freeze()
        self.update_scalarmappable()
        if cbook.is_string_like(self._edgecolors) and self._edgecolors[:2] == 'No':
            self._linewidths = (0,)
            #self._edgecolors = self._facecolors
        renderer.draw_poly_collection(
            self._verts, transform, self.clipbox,
            self._facecolors, self._edgecolors,
            self._linewidths, self._antialiaseds,
            self._offsets,  transoffset)
        transform.thaw()
        transoffset.thaw()
        renderer.close_group('polycollection')


    def get_verts(self, dataTrans=None):
        '''Return vertices in data coordinates.
        The calculation is incomplete in general; it is based
        on the vertices or the offsets, whichever is using
        dataTrans as its transformation, so it does not take
        into account the combined effect of segments and offsets.
        '''
        verts = []
        if self._offsets is None:
            for seg in self._verts:
                verts.extend(seg)
            return [tuple(xy) for xy in verts]
        if self.get_transoffset() == dataTrans:
            return [tuple(xy) for xy in self._offsets]
        raise NotImplementedError('Vertices in data coordinates are calculated\n'
                + 'with offsets only if _transOffset == dataTrans.')

class BrokenBarHCollection(PolyCollection):
    """
    A colleciton of horizontal bars spanning yrange with a sequence of
    xranges
    """
    def __init__(self, xranges, yrange, **kwargs):
        """
        xranges : sequence of (xmin, xwidth)
        yrange  : ymin, ywidth

        %(PatchCollection)s
        """
        ymin, ywidth = yrange
        ymax = ymin + ywidth
        verts = [ [(xmin, ymin), (xmin, ymax), (xmin+xwidth, ymax), (xmin+xwidth, ymin)] for xmin, xwidth in xranges]
        PolyCollection.__init__(self, verts, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class RegularPolyCollection(PatchCollection):
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

        %(PatchCollection)s

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
        PatchCollection.__init__(self,**kwargs)
        self._sizes = sizes
        self._dpi = dpi
        self.numsides = numsides
        self.rotation = rotation
        self._update_verts()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

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


    def _update_verts(self):
        r = 1.0/math.sqrt(math.pi)  # unit area
        theta = (2*math.pi/self.numsides)*npy.arange(self.numsides) + self.rotation
        self._verts = zip( r*npy.sin(theta), r*npy.cos(theta) )

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group('regpolycollection')
        transform = self.get_transform()
        transoffset = self.get_transoffset()

        transform.freeze()
        transoffset.freeze()
        self.update_scalarmappable()
        self._update_verts()
        scales = npy.sqrt(npy.asarray(self._sizes)*self._dpi.get()/72.0)


        offsets = self._offsets
        if self._offsets is not None:
            xs, ys = zip(*offsets)
            #print 'converting: units=%s, converter=%s'%(self.axes.xaxis.units, self.axes.xaxis.converter)
            xs = self.convert_xunits(xs)
            ys = self.convert_yunits(ys)
            offsets = zip(xs, ys)
        else:
            offsets = None

        #print 'drawing offsets', offsets
        #print 'drawing verts', self._verts
        #print 'drawing scales', scales
        if cbook.is_string_like(self._edgecolors) and self._edgecolors[:2] == 'No':
            #self._edgecolors = self._facecolors
            self._linewidths = (0,)
        renderer.draw_regpoly_collection(
            self.clipbox,
            offsets, transoffset,
            self._verts, scales,
            self._facecolors, self._edgecolors,
            self._linewidths, self._antialiaseds)

        transform.thaw()
        transoffset.thaw()
        renderer.close_group('regpolycollection')


    def get_verts(self, dataTrans=None):
        '''Return vertices in data coordinates.
        The calculation is incomplete; it uses only
        the offsets, and only if _transOffset is dataTrans.
        '''
        if self.get_transoffset() == dataTrans:
            return [tuple(xy) for xy in self._offsets]
        raise NotImplementedError('Vertices in data coordinates are calculated\n'
                + 'only with offsets and only if _transOffset == dataTrans.')

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

        %(PatchCollection)s
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

        %(PatchCollection)s
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
                 linestyle = 'solid',
                 offsets = None,
                 transOffset = None,#transforms.identity_transform(),
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

        Collection.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        if linewidths is None   :
            linewidths   = (mpl.rcParams['lines.linewidth'], )

        if colors is None       :
            colors       = (mpl.rcParams['lines.color'],)
        if antialiaseds is None :
            antialiaseds = (mpl.rcParams['lines.antialiased'], )

        self._colors = _colors.colorConverter.to_rgba_list(colors)
        self._aa = self._get_value(antialiaseds)
        self._lw = self._get_value(linewidths)
        self.set_linestyle(linestyle)
        self._uniform_offsets = None
        if offsets is not None:
            offsets = npy.asarray(offsets)
            if len(offsets.shape) == 1:
                offsets = offsets[npy.newaxis,:]  # Make it Nx2.
        if transOffset is None:
            if offsets is not None:
                self._uniform_offsets = offsets
                offsets = None
            transOffset = transforms.identity_transform()
        self._offsets = offsets
        self._transOffset = transOffset
        self.set_segments(segments)
        self.pickradius = pickradius
        self.update(kwargs)

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns T/F, dict(ind=itemlist), where every item in itemlist contains the event.
        """
        import matplotlib.lines as ML
        if callable(self._contains): return self._contains(self,mouseevent)

        # TODO: add offset processing; adjusting the mouse for each offset
        # will be somewhat cheaper than adjusting the segments.
        if self._offsets != None:
            raise NotImplementedError, "LineCollection does not yet support picking with offsets"

        mx,my = mouseevent.x,mouseevent.y
        transform = self.get_transform()

        ind = []
        for this in xrange(len(self._segments)):
            xy = transform.seq_xy_tups(self._segments[this])
            this_ind = ML.segment_hits(mx,my,xy[:,0],xy[:,1],self.pickradius)
            ind.extend([(this,k) for k in this_ind])
        return len(ind)>0,dict(ind=ind)

    def set_pickradius(self,pickradius): self.pickradius = 5

    def get_pickradius(self): return self.pickradius

    def get_transoffset(self):
        if self._transOffset is None:
            self._transOffset = transforms.identity_transform()
        return self._transOffset

    def set_segments(self, segments):
        if segments is None: return
        self._segments = [npy.asarray(seg) for seg in segments]
        if self._uniform_offsets is not None:
            self._add_offsets()

    set_verts = set_segments # for compatibility with PolyCollection

    def _add_offsets(self):
        segs = self._segments
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


    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group('linecollection')
        transform = self.get_transform()
        transoffset = self.get_transoffset()

        transform.freeze()
        transoffset.freeze()

        segments = self._segments
        offsets = self._offsets

        if self.have_units():
            segments = []
            for segment in self._segments:
                xs, ys = zip(*segment)
                xs = self.convert_xunits(xs)
                ys = self.convert_yunits(ys)
                segments.append(zip(xs, ys))
            if self._offsets is not None:
                xs = self.convert_xunits(self._offsets[:0])
                ys = self.convert_yunits(self._offsets[:1])
                offsets = zip(xs, ys)

        self.update_scalarmappable()
        #print 'calling renderer draw line collection'
        renderer.draw_line_collection(
            segments, transform, self.clipbox,
            self._colors, self._lw, self._ls, self._aa, offsets,
            transoffset)
        transform.thaw()
        transoffset.thaw()

        renderer.close_group('linecollection')

    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.  lw can be a scalar or a
        sequence; if it is a sequence the patches will cycle through the
        sequence

        ACCEPTS: float or sequence of floats
        """

        self._lw = self._get_value(lw)
    set_linewidths = set_lw = set_linewidth

    def set_linestyle(self, ls):
        """
        Set the linestyles(s) for the collection.
        ACCEPTS: ['solid' | 'dashed', 'dashdot', 'dotted' |  (offset, on-off-dash-seq) ]
        """
        if cbook.is_string_like(ls):
            dashes = backend_bases.GraphicsContextBase.dashd[ls]
        elif cbook.iterable(ls) and len(ls)==2:
            dashes = ls
        else: raise ValueError('Do not know how to convert %s to dashes'%ls)


        self._ls = dashes

    def set_color(self, c):
        """
        Set the color(s) of the line collection.  c can be a
        matplotlib color arg (all patches have same color), or a a
        sequence or rgba tuples; if it is a sequence the patches will
        cycle through the sequence

        ACCEPTS: matplotlib color arg or sequence of rgba tuples
        """
        self._colors = _colors.colorConverter.to_rgba_list(c)

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

    def set_alpha(self, alpha):
        """
        Set the alpha tranpancies of the collection.  Alpha can be a
        float, in which case it is applied to the entire collection,
        or a sequence of floats

        ACCEPTS: float or sequence of floats
        """

        try: float(alpha)
        except TypeError: raise TypeError('alpha must be a float')
        else:
            artist.Artist.set_alpha(self, alpha)
            self._colors = [(r,g,b,alpha) for r,g,b,a in self._colors]

    def get_linewidth(self):
        return self._lw

    def get_linestyle(self):
        return self._ls

    def get_dashes(self):
        return self._ls

    def get_color(self):
        return self._colors
    get_colors = get_color  # for compatibility with old versions

    def get_verts(self, dataTrans=None):
        '''Return vertices in data coordinates.
        The calculation is incomplete in general; it is based
        on the segments or the offsets, whichever is using
        dataTrans as its transformation, so it does not take
        into account the combined effect of segments and offsets.
        '''
        verts = []
        if self._offsets is None:
            for seg in self._segments:
                verts.extend(seg)
            return [tuple(xy) for xy in verts]
        if self.get_transoffset() == dataTrans:
            return [tuple(xy) for xy in self._offsets]
        raise NotImplementedError('Vertices in data coordinates are calculated\n'
                + 'with offsets only if _transOffset == dataTrans.')

    def update_scalarmappable(self):
        """
        If the scalar mappable array is not none, update colors
        from scalar data
        """
        if self._A is None: return
        if len(self._A.shape)>1:
            raise ValueError('LineCollections can only map rank 1 arrays')
        self._colors = self.to_rgba(self._A, self._alpha)



artist.kwdocd['Collection'] = artist.kwdoc(Collection)
artist.kwdocd['PatchCollection'] = patchstr = artist.kwdoc(PatchCollection)
for k in ('QuadMesh', 'PolyCollection', 'BrokenBarHCollection', 'RegularPolyCollection',
          'StarPolygonCollection'):
    artist.kwdocd[k] = patchstr
artist.kwdocd['LineCollection'] = artist.kwdoc(LineCollection)

