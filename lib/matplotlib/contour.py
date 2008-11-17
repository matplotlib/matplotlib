"""
These are  classes to support contour plotting and
labelling for the axes class
"""
from __future__ import division
import warnings
import matplotlib as mpl
import numpy as np
from numpy import ma
import matplotlib._cntr as _cntr
import matplotlib.path as path
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.collections as collections
import matplotlib.font_manager as font_manager
import matplotlib.text as text
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab

# Import needed for adding manual selection capability to clabel
from matplotlib.blocking_input import BlockingContourLabeler

# We can't use a single line collection for contour because a line
# collection can have only a single line style, and we want to be able to have
# dashed negative contours, for example, and solid positive contours.
# We could use a single polygon collection for filled contours, but it
# seems better to keep line and filled contours similar, with one collection
# per level.


class ContourLabeler:
    '''Mixin to provide labelling capability to ContourSet'''

    def clabel(self, *args, **kwargs):
        """
        call signature::

          clabel(cs, **kwargs)

        adds labels to line contours in *cs*, where *cs* is a
        :class:`~matplotlib.contour.ContourSet` object returned by
        contour.

        ::

          clabel(cs, v, **kwargs)

        only labels contours listed in *v*.

        Optional keyword arguments:

          *fontsize*:
            See http://matplotlib.sf.net/fonts.html

          *colors*:
            - if *None*, the color of each label matches the color of
              the corresponding contour

            - if one string color, e.g. *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color

            - if a tuple of matplotlib color args (string, float, rgb, etc),
              different labels will be plotted in different colors in the order
              specified

          *inline*:
            controls whether the underlying contour is removed or
            not. Default is *True*.

          *inline_spacing*:
            space in pixels to leave on each side of label when
            placing inline.  Defaults to 5.  This spacing will be
            exact for labels at locations where the contour is
            straight, less so for labels on curved contours.

          *fmt*:
            a format string for the label. Default is '%1.3f'
            Alternatively, this can be a dictionary matching contour
            levels with arbitrary strings to use for each contour level
            (i.e., fmt[level]=string)

          *manual*:
            if *True*, contour labels will be placed manually using
            mouse clicks.  Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels.  The third
            button can be used to remove the last label added, but
            only if labels are not inline.  Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

        .. plot:: mpl_examples/pylab_examples/contour_demo.py
        """

        """
        NOTES on how this all works:

        clabel basically takes the input arguments and uses them to
        add a list of "label specific" attributes to the ContourSet
        object.  These attributes are all of the form label* and names
        should be fairly self explanatory.

        Once these attributes are set, clabel passes control to the
        labels method (case of automatic label placement) or
        BlockingContourLabeler (case of manual label placement).
        """

        fontsize = kwargs.get('fontsize', None)
        inline = kwargs.get('inline', 1)
        inline_spacing = kwargs.get('inline_spacing', 5)
        self.labelFmt = kwargs.get('fmt', '%1.3f')
        _colors = kwargs.get('colors', None)

        # Detect if manual selection is desired and remove from argument list
        self.labelManual=kwargs.get('manual',False)

        if len(args) == 0:
            levels = self.levels
            indices = range(len(self.levels))
        elif len(args) == 1:
            levlabs = list(args[0])
            indices, levels = [], []
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                msg = "Specified levels " + str(levlabs)
                msg += "\n don't match available levels "
                msg += str(self.levels)
                raise ValueError(msg)
        else:
            raise TypeError("Illegal arguments to clabel, see help(clabel)")
        self.labelLevelList = levels
        self.labelIndiceList = indices

        self.labelFontProps = font_manager.FontProperties()
        if fontsize == None:
            font_size = int(self.labelFontProps.get_size_in_points())
        else:
            if type(fontsize) not in [int, float, str]:
                raise TypeError("Font size must be an integer number.")
                # Can't it be floating point, as indicated in line above?
            else:
                if type(fontsize) == str:
                    font_size = int(self.labelFontProps.get_size_in_points())
                else:
                    self.labelFontProps.set_size(fontsize)
                    font_size = fontsize
        self.labelFontSizeList = [font_size] * len(levels)

        if _colors == None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = colors.ListedColormap(_colors, N=len(self.labelLevelList))
            self.labelCValueList = range(len(self.labelLevelList))
            self.labelMappable = cm.ScalarMappable(cmap = cmap,
                                                   norm = colors.NoNorm())

        #self.labelTexts = []   # Initialized in ContourSet.__init__
        #self.labelCValues = [] # same
        self.labelXYs = []

        if self.labelManual:
            print 'Select label locations manually using first mouse button.'
            print 'End manual selection with second mouse button.'
            if not inline:
                print 'Remove last label by clicking third mouse button.'

            blocking_contour_labeler = BlockingContourLabeler(self)
            blocking_contour_labeler(inline,inline_spacing)
        else:
            self.labels(inline,inline_spacing)

        # Hold on to some old attribute names.  These are depricated and will
        # be removed in the near future (sometime after 2008-08-01), but keeping
        # for now for backwards compatibility
        self.cl = self.labelTexts
        self.cl_xy = self.labelXYs
        self.cl_cvalues = self.labelCValues

        self.labelTextsList =  cbook.silent_list('text.Text', self.labelTexts)
        return self.labelTextsList


    def print_label(self, linecontour,labelwidth):
        "if contours are too short, don't plot a label"
        lcsize = len(linecontour)
        if lcsize > 10 * labelwidth:
            return 1

        xmax = np.amax(linecontour[:,0])
        xmin = np.amin(linecontour[:,0])
        ymax = np.amax(linecontour[:,1])
        ymin = np.amin(linecontour[:,1])

        lw = labelwidth
        if (xmax - xmin) > 1.2* lw or (ymax - ymin) > 1.2 * lw:
            return 1
        else:
            return 0

    def too_close(self, x,y, lw):
        "if there's a label already nearby, find a better place"
        if self.labelXYs != []:
            dist = [np.sqrt((x-loc[0]) ** 2 + (y-loc[1]) ** 2)
                    for loc in self.labelXYs]
            for d in dist:
                if d < 1.2*lw:
                    return 1
                else: return 0
        else: return 0

    def get_label_coords(self, distances, XX, YY, ysize, lw):
        """ labels are ploted at a location with the smallest
        dispersion of the contour from a straight line
        unless there's another label nearby, in which case
        the second best place on the contour is picked up
        if there's no good place a label isplotted at the
        beginning of the contour
        """

        hysize = int(ysize/2)
        adist = np.argsort(distances)

        for ind in adist:
            x, y = XX[ind][hysize], YY[ind][hysize]
            if self.too_close(x,y, lw):
                continue
            else:
                return x,y, ind

        ind = adist[0]
        x, y = XX[ind][hysize], YY[ind][hysize]
        return x,y, ind

    def get_label_width(self, lev, fmt, fsize):
        "get the width of the label in points"
        if cbook.is_string_like(lev):
            lw = (len(lev)) * fsize
        else:
            lw = (len(self.get_text(lev,fmt))) * fsize

        return lw

    def get_real_label_width( self, lev, fmt, fsize ):
        """
        This computes actual onscreen label width.
        This uses some black magic to determine onscreen extent of non-drawn
        label.  This magic may not be very robust.
        """
        # Find middle of axes
        xx = np.mean( np.asarray(self.ax.axis()).reshape(2,2), axis=1 )

        # Temporarily create text object
        t = text.Text( xx[0], xx[1] )
        self.set_label_props( t, self.get_text(lev,fmt), 'k' )

        # Some black magic to get onscreen extent
        # NOTE: This will only work for already drawn figures, as the canvas
        # does not have a renderer otherwise.  This is the reason this function
        # can't be integrated into the rest of the code.
        bbox = t.get_window_extent(renderer=self.ax.figure.canvas.renderer)

        # difference in pixel extent of image
        lw = np.diff(bbox.corners()[0::2,0])[0]

        return lw

    def set_label_props(self, label, text, color):
        "set the label properties - color, fontsize, text"
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self.labelFontProps)
        label.set_clip_box(self.ax.bbox)

    def get_text(self, lev, fmt):
        "get the text of the label"
        if cbook.is_string_like(lev):
            return lev
        else:
            if isinstance(fmt,dict):
                return fmt[lev]
            else:
                return fmt%lev

    def locate_label(self, linecontour, labelwidth):
        """find a good place to plot a label (relatively flat
        part of the contour) and the angle of rotation for the
        text object
        """

        nsize= len(linecontour)
        if labelwidth > 1:
            xsize = int(np.ceil(nsize/labelwidth))
        else:
            xsize = 1
        if xsize == 1:
            ysize = nsize
        else:
            ysize = labelwidth

        XX = np.resize(linecontour[:,0],(xsize, ysize))
        YY = np.resize(linecontour[:,1],(xsize, ysize))
        #I might have fouled up the following:
        yfirst = YY[:,0].reshape(xsize, 1)
        ylast = YY[:,-1].reshape(xsize, 1)
        xfirst = XX[:,0].reshape(xsize, 1)
        xlast = XX[:,-1].reshape(xsize, 1)
        s = (yfirst-YY) * (xlast-xfirst) - (xfirst-XX) * (ylast-yfirst)
        L = np.sqrt((xlast-xfirst)**2+(ylast-yfirst)**2).ravel()
        dist = np.add.reduce(([(abs(s)[i]/L[i]) for i in range(xsize)]),-1)
        x,y,ind = self.get_label_coords(dist, XX, YY, ysize, labelwidth)
        #print 'ind, x, y', ind, x, y

        # There must be a more efficient way...
        lc = [tuple(l) for l in linecontour]
        dind = lc.index((x,y))
        #print 'dind', dind
        #dind = list(linecontour).index((x,y))

        return x, y, dind

    def calc_label_rot_and_inline( self, slc, ind, lw, lc=None, spacing=5 ):
        """
        This function calculates the appropriate label rotation given
        the linecontour coordinates in screen units, the index of the
        label location and the label width.

        It will also break contour and calculate inlining if *lc* is
        not empty (lc defaults to the empty list if None).  *spacing*
        is the space around the label in pixels to leave empty.

        Do both of these tasks at once to avoid calling mlab.path_length
        multiple times, which is relatively costly.

        The method used here involves calculating the path length
        along the contour in pixel coordinates and then looking
        approximately label width / 2 away from central point to
        determine rotation and then to break contour if desired.
        """

        if lc is None: lc = []
        # Half the label width
        hlw = lw/2.0

        # Check if closed and, if so, rotate contour so label is at edge
        closed = mlab.is_closed_polygon(slc)
        if closed:
            slc = np.r_[ slc[ind:-1], slc[:ind+1] ]

            if len(lc): # Rotate lc also if not empty
                lc = np.r_[ lc[ind:-1], lc[:ind+1] ]

            ind = 0

        # Path length in pixel space
        pl = mlab.path_length(slc)
        pl = pl-pl[ind]

        # Use linear interpolation to get points around label
        xi = np.array( [ -hlw, hlw ] )
        if closed: # Look at end also for closed contours
            dp = np.array([pl[-1],0])
        else:
            dp = np.zeros_like(xi)

        ll = mlab.less_simple_linear_interpolation( pl, slc, dp+xi,
                                                     extrap=True )

        # get vector in pixel space coordinates from one point to other
        dd = np.diff( ll, axis=0 ).ravel()

        # Get angle of vector - must be calculated in pixel space for
        # text rotation to work correctly
        if np.all(dd==0): # Must deal with case of zero length label
            rotation = 0.0
        else:
            rotation = np.arctan2(dd[1], dd[0]) * 180.0 / np.pi

        # Fix angle so text is never upside-down
        if rotation > 90:
            rotation = rotation - 180.0
        if rotation < -90:
            rotation = 180.0 + rotation

        # Break contour if desired
        nlc = []
        if len(lc):
            # Expand range by spacing
            xi = dp + xi + np.array([-spacing,spacing])

            # Get indices near points of interest
            I = mlab.less_simple_linear_interpolation(
                pl, np.arange(len(pl)), xi, extrap=False )

            # If those indices aren't beyond contour edge, find x,y
            if (not np.isnan(I[0])) and int(I[0])<>I[0]:
                xy1 = mlab.less_simple_linear_interpolation(
                    pl, lc, [ xi[0] ] )

            if (not np.isnan(I[1])) and int(I[1])<>I[1]:
                xy2 = mlab.less_simple_linear_interpolation(
                    pl, lc, [ xi[1] ] )

            # Make integer
            I = [ np.floor(I[0]), np.ceil(I[1]) ]

            # Actually break contours
            if closed:
                # This will remove contour if shorter than label
                if np.all(~np.isnan(I)):
                    nlc.append( np.r_[ xy2, lc[I[1]:I[0]+1], xy1 ] )
            else:
                # These will remove pieces of contour if they have length zero
                if not np.isnan(I[0]):
                    nlc.append( np.r_[ lc[:I[0]+1], xy1 ] )
                if not np.isnan(I[1]):
                    nlc.append( np.r_[ xy2, lc[I[1]:] ] )

            # The current implementation removes contours completely
            # covered by labels.  Uncomment line below to keep
            # original contour if this is the preferred behavoir.
            #if not len(nlc): nlc = [ lc ]

        return (rotation,nlc)


    def add_label(self,x,y,rotation,lev,cvalue):
        dx,dy = self.ax.transData.inverted().transform_point((x,y))
        t = text.Text(dx, dy, rotation = rotation,
                      horizontalalignment='center',
                      verticalalignment='center')

        color = self.labelMappable.to_rgba(cvalue,alpha=self.alpha)

        _text = self.get_text(lev,self.labelFmt)
        self.set_label_props(t, _text, color)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x,y))

        # Add label to plot here - useful for manual mode label selection
        self.ax.add_artist(t)

    def pop_label(self,index=-1):
        '''Defaults to removing last label, but any index can be supplied'''
        self.labelCValues.pop(index)
        t = self.labelTexts.pop(index)
        t.remove()

    def labels(self, inline, inline_spacing):
        trans = self.ax.transData # A bit of shorthand

        for icon, lev, fsize, cvalue in zip(
            self.labelIndiceList, self.labelLevelList, self.labelFontSizeList,
            self.labelCValueList ):

            con = self.collections[icon]
            lw = self.get_label_width(lev, self.labelFmt, fsize)
            additions = []
            paths = con.get_paths()
            for segNum, linepath in enumerate(paths):
                lc = linepath.vertices # Line contour
                slc0 = trans.transform(lc) # Line contour in screen coords

                # For closed polygons, add extra point to avoid division by
                # zero in print_label and locate_label.  Other than these
                # functions, this is not necessary and should probably be
                # eventually removed.
                if mlab.is_closed_polygon( lc ):
                    slc = np.r_[ slc0, slc0[1:2,:] ]
                else:
                    slc = slc0

                if self.print_label(slc,lw): # Check if long enough for a label
                    x,y,ind  = self.locate_label(slc, lw)

                    if inline: lcarg = lc
                    else: lcarg = None
                    rotation,new=self.calc_label_rot_and_inline(
                        slc0, ind, lw, lcarg,
                        inline_spacing )

                    # Actually add the label
                    self.add_label(x,y,rotation,lev,cvalue)

                    # If inline, add new contours
                    if inline:
                        for n in new:
                            # Add path if not empty or single point
                            if len(n)>1: additions.append( path.Path(n) )
                else: # If not adding label, keep old path
                    additions.append(linepath)

            # After looping over all segments on a contour, remove old
            # paths and add new ones if inlining
            if inline:
                del paths[:]
                paths.extend(additions)

class ContourSet(cm.ScalarMappable, ContourLabeler):
    """
    Create and store a set of contour lines or filled regions.

    User-callable method: clabel

    Useful attributes:
      ax:
        the axes object in which the contours are drawn
      collections:
        a silent_list of LineCollections or PolyCollections
      levels:
        contour levels
      layers:
        same as levels for line contours; half-way between
        levels for filled contours.  See _process_colors method.
    """


    def __init__(self, ax, *args, **kwargs):
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg 'filled' is False (default) or True.

        The first argument of the initializer must be an axes
        object.  The remaining arguments and keyword arguments
        are described in ContourSet.contour_doc.

        """
        self.ax = ax
        self.levels = kwargs.get('levels', None)
        self.filled = kwargs.get('filled', False)
        self.linewidths = kwargs.get('linewidths', None)
        self.linestyles = kwargs.get('linestyles', 'solid')

        self.alpha = kwargs.get('alpha', 1.0)
        self.origin = kwargs.get('origin', None)
        self.extent = kwargs.get('extent', None)
        cmap = kwargs.get('cmap', None)
        self.colors = kwargs.get('colors', None)
        norm = kwargs.get('norm', None)
        self.extend = kwargs.get('extend', 'neither')
        self.antialiased = kwargs.get('antialiased', True)
        self.nchunk = kwargs.get('nchunk', 0)
        self.locator = kwargs.get('locator', None)
        if (isinstance(norm, colors.LogNorm)
                or isinstance(self.locator, ticker.LogLocator)):
            self.logscale = True
            if norm is None:
                norm = colors.LogNorm()
            if self.extend is not 'neither':
                raise ValueError('extend kwarg does not work yet with log scale')
        else:
            self.logscale = False

        if self.origin is not None: assert(self.origin in
                                            ['lower', 'upper', 'image'])
        if self.extent is not None: assert(len(self.extent) == 4)
        if cmap is not None: assert(isinstance(cmap, colors.Colormap))
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image': self.origin = mpl.rcParams['image.origin']
        x, y, z = self._contour_args(*args)        # also sets self.levels,
                                                   #  self.layers
        if self.colors is not None:
            cmap = colors.ListedColormap(self.colors, N=len(self.layers))
        if self.filled:
            self.collections = cbook.silent_list('collections.PolyCollection')
        else:
            self.collections = cbook.silent_list('collections.LineCollection')
        # label lists must be initialized here
        self.labelTexts = []
        self.labelCValues = []

        kw = {'cmap': cmap}
        if norm is not None:
            kw['norm'] = norm
        cm.ScalarMappable.__init__(self, **kw) # sets self.cmap;
        self._process_colors()
        _mask = ma.getmask(z)
        if _mask is ma.nomask:
            _mask = None

        if self.filled:
            if self.linewidths is not None:
                warnings.warn('linewidths is ignored by contourf')
            C = _cntr.Cntr(x, y, z.filled(), _mask)
            lowers = self._levels[:-1]
            uppers = self._levels[1:]
            for level, level_upper in zip(lowers, uppers):
                nlist = C.trace(level, level_upper, points = 0,
                        nchunk = self.nchunk)
                col = collections.PolyCollection(nlist,
                                     antialiaseds = (self.antialiased,),
                                     edgecolors= 'none',
                                     alpha=self.alpha)
                self.ax.add_collection(col)
                self.collections.append(col)

        else:
            tlinewidths = self._process_linewidths()
            self.tlinewidths = tlinewidths
            tlinestyles = self._process_linestyles()
            C = _cntr.Cntr(x, y, z.filled(), _mask)
            for level, width, lstyle in zip(self.levels, tlinewidths, tlinestyles):
                nlist = C.trace(level, points = 0)
                col = collections.LineCollection(nlist,
                                     linewidths = width,
                                     linestyle = lstyle,
                                     alpha=self.alpha)

                if level < 0.0 and self.monochrome:
                    ls = mpl.rcParams['contour.negative_linestyle']
                    col.set_linestyle(ls)
                col.set_label('_nolegend_')
                self.ax.add_collection(col, False)
                self.collections.append(col)
        self.changed() # set the colors
        x0 = ma.minimum(x)
        x1 = ma.maximum(x)
        y0 = ma.minimum(y)
        y1 = ma.maximum(y)
        self.ax.update_datalim([(x0,y0), (x1,y1)])
        self.ax.autoscale_view()

    def changed(self):
        tcolors = [ (tuple(rgba),) for rgba in
                                self.to_rgba(self.cvalues, alpha=self.alpha)]
        self.tcolors = tcolors
        for color, collection in zip(tcolors, self.collections):
            collection.set_alpha(self.alpha)
            collection.set_color(color)
        for label, cv in zip(self.labelTexts, self.labelCValues):
            label.set_alpha(self.alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        # add label colors
        cm.ScalarMappable.changed(self)


    def _autolev(self, z, N):
        '''
        Select contour levels to span the data.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        '''
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N+1)
        self.locator.create_dummy_axis()
        zmax = self.zmax
        zmin = self.zmin
        self.locator.set_bounds(zmin, zmax)
        lev = self.locator()
        zmargin = (zmax - zmin) * 0.000001 # so z < (zmax + zmargin)
        if zmax >= lev[-1]:
            lev[-1] += zmargin
        if zmin <= lev[0]:
            if self.logscale:
                lev[0] = 0.99 * zmin
            else:
                lev[0] -= zmargin
        self._auto = True
        if self.filled:
            return lev
        return lev[1:-1]

    def _initialize_x_y(self, z):
        '''
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i,j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        '''
        if z.ndim != 2:
            raise TypeError("Input must be a 2D array.")
        else:
            Ny, Nx = z.shape
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0,x1,y0,y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0,x1,y0,y1 = (0, Nx, 0, Ny)
        else:
            x0,x1,y0,y1 = self.extent
        dx = float(x1 - x0)/Nx
        dy = float(y1 - y0)/Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x,y)

    def _check_xyz(self, args):
        '''
        For functions like contour, check that the dimensions
        of the input arrays match; if x and y are 1D, convert
        them to 2D using meshgrid.

        Possible change: I think we should make and use an ArgumentError
        Exception class (here and elsewhere).
        '''
        # We can strip away the x and y units
        x = self.ax.convert_xunits( args[0] )
        y = self.ax.convert_yunits( args[1] )

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = ma.asarray(args[2], dtype=np.float64)
        if z.ndim != 2:
            raise TypeError("Input z must be a 2D array.")
        else: Ny, Nx = z.shape
        if x.shape == z.shape and y.shape == z.shape:
            return x,y,z
        if x.ndim != 1 or y.ndim != 1:
            raise TypeError("Inputs x and y must be 1D or 2D.")
        nx, = x.shape
        ny, = y.shape
        if nx != Nx or ny != Ny:
            raise TypeError("Length of x must be number of columns in z,\n" +
                            "and length of y must be number of rows.")
        x,y = np.meshgrid(x,y)
        return x,y,z



    def _contour_args(self, *args):
        if self.filled: fn = 'contourf'
        else:           fn = 'contour'
        Nargs = len(args)
        if Nargs <= 2:
            z = ma.asarray(args[0], dtype=np.float64)
            x, y = self._initialize_x_y(z)
        elif Nargs <=4:
            x,y,z = self._check_xyz(args[:3])
        else:
            raise TypeError("Too many arguments to %s; see help(%s)" % (fn,fn))
        self.zmax = ma.maximum(z)
        self.zmin = ma.minimum(z)
        if self.logscale and self.zmin <= 0:
            z = ma.masked_where(z <= 0, z)
            warnings.warn('Log scale: values of z <=0 have been masked')
            self.zmin = z.min()
        self._auto = False
        if self.levels is None:
            if Nargs == 1 or Nargs == 3:
                lev = self._autolev(z, 7)
            else:   # 2 or 4 args
                level_arg = args[-1]
                try:
                    if type(level_arg) == int:
                        lev = self._autolev(z, level_arg)
                    else:
                        lev = np.asarray(level_arg).astype(np.float64)
                except:
                    raise TypeError(
                        "Last %s arg must give levels; see help(%s)" % (fn,fn))
            if self.filled and len(lev) < 2:
                raise ValueError("Filled contours require at least 2 levels.")
            # Workaround for cntr.c bug wrt masked interior regions:
            #if filled:
            #    z = ma.masked_array(z.filled(-1e38))
            # It's not clear this is any better than the original bug.
            self.levels = lev
        #if self._auto and self.extend in ('both', 'min', 'max'):
        #    raise TypeError("Auto level selection is inconsistent "
        #                             + "with use of 'extend' kwarg")
        self._levels = list(self.levels)
        if self.extend in ('both', 'min'):
            self._levels.insert(0, min(self.levels[0],self.zmin) - 1)
        if self.extend in ('both', 'max'):
            self._levels.append(max(self.levels[-1],self.zmax) + 1)
        self._levels = np.asarray(self._levels)
        self.vmin = np.amin(self.levels)  # alternative would be self.layers
        self.vmax = np.amax(self.levels)
        if self.extend in ('both', 'min'):
            self.vmin = 2 * self.levels[0] - self.levels[1]
        if self.extend in ('both', 'max'):
            self.vmax = 2 * self.levels[-1] - self.levels[-2]
        self.layers = self._levels # contour: a line is a thin layer
        if self.filled:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])
            if self.extend in ('both', 'min'):
                self.layers[0] = 0.5 * (self.vmin + self._levels[1])
            if self.extend in ('both', 'max'):
                self.layers[-1] = 0.5 * (self.vmax + self._levels[-2])

        return (x, y, z)

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the color mapping on the contour levels,
        not on the actual range of the Z values.  This means we
        don't have to worry about bad values in Z, and we always have
        the full dynamic range available for the selected levels.

        The color is based on the midpoint of the layer, except for
        an extended end layers.
        """
        self.monochrome = self.cmap.monochrome
        if self.colors is not None:
            i0, i1 = 0, len(self.layers)
            if self.extend in ('both', 'min'):
                i0 = -1
            if self.extend in ('both', 'max'):
                i1 = i1 + 1
            self.cvalues = range(i0, i1)
            self.set_norm(colors.NoNorm())
        else:
            self.cvalues = self.layers
        if not self.norm.scaled():
            self.set_clim(self.vmin, self.vmax)
        if self.extend in ('both', 'max', 'min'):
            self.norm.clip = False
        self.set_array(self.layers)
        # self.tcolors are set by the "changed" method

    def _process_linewidths(self):
        linewidths = self.linewidths
        Nlev = len(self.levels)
        if linewidths is None:
            tlinewidths = [(mpl.rcParams['lines.linewidth'],)] *Nlev
        else:
            if cbook.iterable(linewidths) and len(linewidths) < Nlev:
                linewidths = list(linewidths) * int(np.ceil(Nlev/len(linewidths)))
            elif not cbook.iterable(linewidths) and type(linewidths) in [int, float]:
                linewidths = [linewidths] * Nlev
            tlinewidths = [(w,) for w in linewidths]
        return tlinewidths

    def _process_linestyles(self):
        linestyles = self.linestyles
        Nlev = len(self.levels)
        if linestyles is None:
            tlinestyles = ['solid'] * Nlev
        else:
            if cbook.is_string_like(linestyles):
                tlinestyles = [linestyles] * Nlev
            elif cbook.iterable(linestyles) and len(linestyles) <= Nlev:
                tlinestyles = list(linestyles) * int(np.ceil(Nlev/len(linestyles)))
        return tlinestyles

    def get_alpha(self):
        '''returns alpha to be applied to all ContourSet artists'''
        return self.alpha

    def set_alpha(self, alpha):
        '''sets alpha for all ContourSet artists'''
        self.alpha = alpha
        self.changed()

    contour_doc = """
        :func:`~matplotlib.pyplot.contour` and
        :func:`~matplotlib.pyplot.contourf` draw contour lines and
        filled contours, respectively.  Except as noted, function
        signatures and return values are the same for both versions.

        :func:`~matplotlib.pyplot.contourf` differs from the Matlab
        (TM) version in that it does not draw the polygon edges,
        because the contouring engine yields simply connected regions
        with branch cuts.  To draw the edges, add line contours with
        calls to :func:`~matplotlib.pyplot.contour`.


        call signatures::

          contour(Z)

        make a contour plot of an array *Z*. The level values are chosen
        automatically.

        ::

          contour(X,Y,Z)

        *X*, *Y* specify the (*x*, *y*) coordinates of the surface

        ::

          contour(Z,N)
          contour(X,Y,Z,N)

        contour *N* automatically-chosen levels.

        ::

          contour(Z,V)
          contour(X,Y,Z,V)

        draw contour lines at the values specified in sequence *V*

        ::

          contourf(..., V)

        fill the (len(*V*)-1) regions between the values in *V*

        ::

          contour(Z, **kwargs)

        Use keyword args to control colors, linewidth, origin, cmap ... see
        below for more details.

        *X*, *Y*, and *Z* must be arrays with the same dimensions.

        *Z* may be a masked array, but filled contouring may not
        handle internal masked regions correctly.

        ``C = contour(...)`` returns a
        :class:`~matplotlib.contour.ContourSet` object.

        Optional keyword arguments:

          *colors*: [ None | string | (mpl_colors) ]
            If *None*, the colormap specified by cmap will be used.

            If a string, like 'r' or 'red', all levels will be plotted in this
            color.

            If a tuple of matplotlib color args (string, float, rgb, etc),
            different levels will be plotted in different colors in the order
            specified.

          *alpha*: float
            The alpha blending value

          *cmap*: [ None | Colormap ]
            A cm :class:`~matplotlib.cm.Colormap` instance or
            *None*. If *cmap* is *None* and *colors* is *None*, a
            default Colormap is used.

          *norm*: [ None | Normalize ]
            A :class:`matplotlib.colors.Normalize` instance for
            scaling data values to colors. If *norm* is *None* and
            *colors* is *None*, the default linear scaling is used.

          *origin*: [ None | 'upper' | 'lower' | 'image' ]
            If *None*, the first value of *Z* will correspond to the
            lower left corner, location (0,0). If 'image', the rc
            value for ``image.origin`` will be used.

            This keyword is not active if *X* and *Y* are specified in
            the call to contour.

          *extent*: [ None | (x0,x1,y0,y1) ]

            If *origin* is not *None*, then *extent* is interpreted as
            in :func:`matplotlib.pyplot.imshow`: it gives the outer
            pixel boundaries. In this case, the position of Z[0,0]
            is the center of the pixel, not a corner. If *origin* is
            *None*, then (*x0*, *y0*) is the position of Z[0,0], and
            (*x1*, *y1*) is the position of Z[-1,-1].

            This keyword is not active if *X* and *Y* are specified in
            the call to contour.

          *locator*: [ None | ticker.Locator subclass ]
            If *locator* is None, the default
            :class:`~matplotlib.ticker.MaxNLocator` is used. The
            locator is used to determine the contour levels if they
            are not given explicitly via the *V* argument.

          *extend*: [ 'neither' | 'both' | 'min' | 'max' ]
            Unless this is 'neither', contour levels are automatically
            added to one or both ends of the range so that all data
            are included. These added ranges are then mapped to the
            special colormap values which default to the ends of the
            colormap range, but can be set via
            :meth:`matplotlib.cm.Colormap.set_under` and
            :meth:`matplotlib.cm.Colormap.set_over` methods.

        contour-only keyword arguments:

          *linewidths*: [ None | number | tuple of numbers ]
            If *linewidths* is *None*, the default width in
            ``lines.linewidth`` in ``matplotlibrc`` is used.

            If a number, all levels will be plotted with this linewidth.

            If a tuple, different levels will be plotted with different
            linewidths in the order specified

          *linestyles*: [None | 'solid' | 'dashed' | 'dashdot' | 'dotted' ]
            If *linestyles* is *None*, the 'solid' is used.

            *linestyles* can also be an iterable of the above strings
            specifying a set of linestyles to be used. If this
            iterable is shorter than the number of contour levels
            it will be repeated as necessary.

            If contour is using a monochrome colormap and the contour
            level is less than 0, then the linestyle specified
            in ``contour.negative_linestyle`` in ``matplotlibrc``
            will be used.

        contourf-only keyword arguments:

          *antialiased*: [ True | False ]
            enable antialiasing

          *nchunk*: [ 0 | integer ]
            If 0, no subdivision of the domain. Specify a positive integer to
            divide the domain into subdomains of roughly *nchunk* by *nchunk*
            points. This may never actually be advantageous, so this option may
            be removed. Chunking introduces artifacts at the chunk boundaries
            unless *antialiased* is *False*.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/contour_demo.py
        """

    def find_nearest_contour( self, x, y, indices=None, pixel=True ):
        """
        Finds contour that is closest to a point.  Defaults to
        measuring distance in pixels (screen space - useful for manual
        contour labeling), but this can be controlled via a keyword
        argument.

        Returns a tuple containing the contour, segment, index of
        segment, x & y of segment point and distance to minimum point.

        Call signature::

          conmin,segmin,imin,xmin,ymin,dmin = find_nearest_contour(
                     self, x, y, indices=None, pixel=True )

        Optional keyword arguments::

        *indices*:
           Indexes of contour levels to consider when looking for
           nearest point.  Defaults to using all levels.

        *pixel*:
           If *True*, measure distance in pixel space, if not, measure
           distance in axes space.  Defaults to *True*.

        """

        # This function uses a method that is probably quite
        # inefficient based on converting each contour segment to
        # pixel coordinates and then comparing the given point to
        # those coordinates for each contour.  This will probably be
        # quite slow for complex contours, but for normal use it works
        # sufficiently well that the time is not noticeable.
        # Nonetheless, improvements could probably be made.

        if indices==None:
            indices = range(len(self.levels))

        dmin = 1e10
        conmin = None
        segmin = None
        xmin = None
        ymin = None

        for icon in indices:
            con = self.collections[icon]
            paths = con.get_paths()
            for segNum, linepath in enumerate(paths):
                lc = linepath.vertices

                # transfer all data points to screen coordinates if desired
                if pixel:
                    lc = self.ax.transData.transform(lc)

                ds = (lc[:,0]-x)**2 + (lc[:,1]-y)**2
                d = min( ds )
                if d < dmin:
                    dmin = d
                    conmin = icon
                    segmin = segNum
                    imin = mpl.mlab.find( ds == d )[0]
                    xmin = lc[imin,0]
                    ymin = lc[imin,1]

        return (conmin,segmin,imin,xmin,ymin,dmin)

