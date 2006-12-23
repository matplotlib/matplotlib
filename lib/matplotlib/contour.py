"""
These are  classes to support contour plotting and
labelling for the axes class
"""
from __future__ import division
from matplotlib import rcParams
import numerix.ma as ma

from ticker import MaxNLocator
from transforms import Interval, Value
from numerix import absolute, arange, array, asarray, ones, divide,\
     transpose, log, log10, Float, Float32, ravel, zeros, Int16,\
     Int32, Int, Float64, ceil, indices, shape, which, where, sqrt,\
     asum, resize, reshape, add, argmin, arctan2, pi, argsort, sin,\
     cos, nonzero, take, concatenate, all, newaxis

from mlab import linspace, meshgrid
import _contour
from cm import ScalarMappable
from cbook import iterable, is_string_like, flatten, enumerate, \
     allequal, dict_delall, strip_math, popd, popall, silent_list
from colors import colorConverter, Normalize, Colormap, ListedColormap, NoNorm
from collections import  PolyCollection, LineCollection
from font_manager import FontProperties
from numerix.mlab import amin, amax
from text import Text
import warnings

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
        clabel(CS, **kwargs) - add labels to line contours in CS,
               where CS is a ContourSet object returned by contour.

        clabel(CS, V, **kwargs) - only label contours listed in V

        keyword arguments:

        * fontsize = None: as described in http://matplotlib.sf.net/fonts.html

        * colors = None:

           - a tuple of matplotlib color args (string, float, rgb, etc),
             different labels will be plotted in different colors in the order
             specified

           - one string color, e.g. colors = 'r' or colors = 'red', all labels
             will be plotted in this color

           - if colors == None, the color of each label matches the color
             of the corresponding contour

        * inline = True: controls whether the underlying contour is removed
                     (inline = True) or not (False)

        * fmt = '%1.3f': a format string for the label

        """
        fontsize = kwargs.get('fontsize', None)
        inline = kwargs.get('inline', 1)
        self.fmt = kwargs.get('fmt', '%1.3f')
        colors = kwargs.get('colors', None)



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
        self.label_levels = levels
        self.label_indices = indices

        self.fp = FontProperties()
        if fontsize == None:
            font_size = int(self.fp.get_size_in_points())
        else:
            if type(fontsize) not in [int, float, str]:
                raise TypeError("Font size must be an integer number.")
                # Can't it be floating point, as indicated in line above?
            else:
                if type(fontsize) == str:
                    font_size = int(self.fp.get_size_in_points())
                else:
                    self.fp.set_size(fontsize)
                    font_size = fontsize
        self.fslist = [font_size] * len(levels)

        if colors == None:
            self.label_mappable = self
            self.label_cvalues = take(self.cvalues, self.label_indices)
        else:
            cmap = ListedColormap(colors, N=len(self.label_levels))
            self.label_cvalues = range(len(self.label_levels))
            self.label_mappable = ScalarMappable(cmap = cmap,
                                                 norm = NoNorm())

        #self.cl = []   # Initialized in ContourSet.__init__
        #self.cl_cvalues = [] # same
        self.cl_xy = []

        self.labels(inline)

        for label in self.cl:
            self.ax.add_artist(label)

        self.label_list =  silent_list('Text', self.cl)
        return self.label_list


    def print_label(self, linecontour,labelwidth):
        "if contours are too short, don't plot a label"
        lcsize = len(linecontour)
        if lcsize > 10 * labelwidth:
            return 1

        xmax = amax(array(linecontour)[:,0])
        xmin = amin(array(linecontour)[:,0])
        ymax = amax(array(linecontour)[:,1])
        ymin = amin(array(linecontour)[:,1])

        lw = labelwidth
        if (xmax - xmin) > 1.2* lw or (ymax - ymin) > 1.2 * lw:
            return 1
        else:
            return 0

    def too_close(self, x,y, lw):
        "if there's a label already nearby, find a better place"
        if self.cl_xy != []:
            dist = [sqrt((x-loc[0]) ** 2 + (y-loc[1]) ** 2)
                    for loc in self.cl_xy]
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
        adist = argsort(distances)

        for ind in adist:
            x, y = XX[ind][hysize], YY[ind][hysize]
            if self.too_close(x,y, lw):
                continue
            else:
                self.cl_xy.append((x,y))
                return x,y, ind

        ind = adist[0]
        x, y = XX[ind][hysize], YY[ind][hysize]
        self.cl_xy.append((x,y))
        return x,y, ind

    def get_label_width(self, lev, fmt, fsize):
        "get the width of the label in points"
        if is_string_like(lev):
            lw = (len(lev)) * fsize
        else:
            lw = (len(fmt%lev)) * fsize

        return lw


    def set_label_props(self, label, text, color):
        "set the label properties - color, fontsize, text"
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self.fp)
        label.set_clip_box(self.ax.bbox)

    def get_text(self, lev, fmt):
        "get the text of the label"
        if is_string_like(lev):
            return lev
        else:
            return fmt%lev


    def break_linecontour(self, linecontour, rot, labelwidth, ind):
        "break a contour in two contours at the location of the label"
        lcsize = len(linecontour)
        hlw = int(labelwidth/2)

        #length of label in screen coords
        ylabel = abs(hlw * sin(rot*pi/180))
        xlabel = abs(hlw * cos(rot*pi/180))

        trans = self.ax.transData

        slc = trans.seq_xy_tups(linecontour)
        x,y = slc[ind]
        xx= asarray(slc)[:,0].copy()
        yy=asarray(slc)[:,1].copy()

        #indices which are under the label
        inds=nonzero(((xx < x+xlabel) & (xx > x-xlabel)) &
                     ((yy < y+ylabel) & (yy > y-ylabel)))

        if len(inds) >0:
            #if the label happens to be over the beginning of the
            #contour, the entire contour is removed, i.e.
            #indices to be removed are
            #inds= [0,1,2,3,305,306,307]
            #should rewrite this in a better way
            linds = nonzero(inds[1:]- inds[:-1] != 1)
            if inds[0] == 0 and len(linds) != 0:
                ii = inds[linds[0]]
                lc1 =linecontour[ii+1:inds[ii+1]]
                lc2 = []

            else:
                lc1=linecontour[:inds[0]]
                lc2= linecontour[inds[-1]+1:]

        else:
            lc1=linecontour[:ind]
            lc2 = linecontour[ind+1:]


        if rot <0:
            new_x1, new_y1 = x-xlabel, y+ylabel
            new_x2, new_y2 = x+xlabel, y-ylabel
        else:
            new_x1, new_y1 = x-xlabel, y-ylabel
            new_x2, new_y2 = x+xlabel, y+ylabel

        new_x1d, new_y1d = trans.inverse_xy_tup((new_x1, new_y1))
        new_x2d, new_y2d = trans.inverse_xy_tup((new_x2, new_y2))
        new_xy1 = array(((new_x1d, new_y1d),))
        new_xy2 = array(((new_x2d, new_y2d),))


        if rot > 0:
            if (len(lc1) > 0 and (lc1[-1][0] <= new_x1d)
                             and (lc1[-1][1] <= new_y1d)):
                lc1 = concatenate((lc1, new_xy1))
                #lc1.append((new_x1d, new_y1d))

            if (len(lc2) > 0 and (lc2[0][0] >= new_x2d)
                             and (lc2[0][1] >= new_y2d)):
                lc2 = concatenate((new_xy2, lc2))
                #lc2.insert(0, (new_x2d, new_y2d))
        else:
            if (len(lc1) > 0 and ((lc1[-1][0] <= new_x1d)
                             and (lc1[-1][1] >= new_y1d))):
                lc1 = concatenate((lc1, new_xy1))
                #lc1.append((new_x1d, new_y1d))

            if (len(lc2) > 0 and ((lc2[0][0] >= new_x2d)
                             and (lc2[0][1] <= new_y2d))):
                lc2 = concatenate((new_xy2, lc2))
                #lc2.insert(0, (new_x2d, new_y2d))

        return [lc1,lc2]


    def locate_label(self, linecontour, labelwidth):
        """find a good place to plot a label (relatively flat
        part of the contour) and the angle of rotation for the
        text object
        """

        nsize= len(linecontour)
        if labelwidth > 1:
            xsize = int(ceil(nsize/labelwidth))
        else:
            xsize = 1
        if xsize == 1:
            ysize = nsize
        else:
            ysize = labelwidth

        XX = resize(asarray(linecontour)[:,0],(xsize, ysize))
        YY = resize(asarray(linecontour)[:,1],(xsize,ysize))

        yfirst = YY[:,0]
        ylast = YY[:,-1]
        xfirst = XX[:,0]
        xlast = XX[:,-1]
        s = ( (reshape(yfirst, (xsize,1))-YY) *
              (reshape(xlast,(xsize,1)) - reshape(xfirst,(xsize,1)))
              - (reshape(xfirst,(xsize,1))-XX)
              * (reshape(ylast,(xsize,1)) - reshape(yfirst,(xsize,1))) )
        L=sqrt((xlast-xfirst)**2+(ylast-yfirst)**2)
        dist = add.reduce(([(abs(s)[i]/L[i]) for i in range(xsize)]),-1)
        x,y,ind = self.get_label_coords(dist, XX, YY, ysize, labelwidth)
        #print 'ind, x, y', ind, x, y
        angle = arctan2(ylast - yfirst, xlast - xfirst)
        rotation = angle[ind]*180/pi
        if rotation > 90:
            rotation = rotation -180
        if rotation < -90:
            rotation = 180 + rotation

        # There must be a more efficient way...
        lc = [tuple(l) for l in linecontour]
        dind = lc.index((x,y))
        #print 'dind', dind
        #dind = list(linecontour).index((x,y))

        return x,y, rotation, dind

    def labels(self, inline):
        levels = self.label_levels
        fslist = self.fslist
        trans = self.ax.transData
        colors = self.label_mappable.to_rgba(self.label_cvalues)
        fmt = self.fmt
        for icon, lev, color, cvalue, fsize in zip(self.label_indices,
                                          self.label_levels,
                                          colors,
                                          self.label_cvalues, fslist):
            con = self.collections[icon]
            lw = self.get_label_width(lev, fmt, fsize)
            additions = []
            for segNum, linecontour in enumerate(con._segments):
                # for closed contours add one more point to
                # avoid division by zero
                if all(linecontour[0] == linecontour[-1]):
                    linecontour = concatenate((linecontour,
                                               linecontour[1][newaxis,:]))
                    #linecontour.append(linecontour[1])
                # transfer all data points to screen coordinates
                slc = trans.seq_xy_tups(linecontour)
                if self.print_label(slc,lw):
                    x,y, rotation, ind  = self.locate_label(slc, lw)
                    # transfer the location of the label back to
                    # data coordinates
                    dx,dy = trans.inverse_xy_tup((x,y))
                    t = Text(dx, dy, rotation = rotation,
                             horizontalalignment='center',
                             verticalalignment='center')
                    text = self.get_text(lev,fmt)
                    self.set_label_props(t, text, color)
                    self.cl.append(t)
                    self.cl_cvalues.append(cvalue)
                    if inline:
                        new = self.break_linecontour(linecontour, rotation,
                                                       lw, ind)
                        con._segments[segNum] = new[0]
                        additions.append(new[1])
            con._segments.extend(additions)

class ContourSet(ScalarMappable, ContourLabeler):
    """
    Create and store a set of contour lines or filled regions.

    User-callable method: clabel

    Useful attributes:
        ax - the axes object in which the contours are drawn
        collections - a silent_list of LineCollections or PolyCollections
        levels - contour levels
        layers - same as levels for line contours; half-way between
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

        self.alpha = kwargs.get('alpha', 1.0)
        self.origin = kwargs.get('origin', None)
        self.extent = kwargs.get('extent', None)
        cmap = kwargs.get('cmap', None)
        self.colors = kwargs.get('colors', None)
        norm = kwargs.get('norm', None)
        self.clip_ends = kwargs.get('clip_ends', None)      ########
        self.extend = kwargs.get('extend', 'neither')
        if self.clip_ends is not None:
            warnings.warn("'clip_ends' has been replaced by 'extend'")
            self.levels = self.levels[1:-1] # discard specified end levels
            self.extend = 'both'            # regenerate end levels
        self.antialiased = kwargs.get('antialiased', True)
        self.nchunk = kwargs.get('nchunk', 0)
        self.locator = kwargs.get('locator', None)

        if self.origin is not None: assert(self.origin in
                                            ['lower', 'upper', 'image'])
        if self.extent is not None: assert(len(self.extent) == 4)
        if cmap is not None: assert(isinstance(cmap, Colormap))
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image': self.origin = rcParams['image.origin']
        x, y, z = self._contour_args(*args)        # also sets self.levels,
                                                   #  self.layers
        if self.colors is not None:
            cmap = ListedColormap(self.colors, N=len(self.layers))
        if self.filled:
            self.collections = silent_list('PolyCollection')
        else:
            self.collections = silent_list('LineCollection')
        # label lists must be initialized here
        self.cl = []
        self.cl_cvalues = []

        kw = {'cmap': cmap}
        if norm is not None:
            kw['norm'] = norm
        ScalarMappable.__init__(self, **kw) # sets self.cmap;
        self._process_colors()

        if self.filled:
            if self.linewidths is None:
                self.linewidths = 0.05 # Good default for Postscript.
            if iterable(self.linewidths):
                self.linewidths = self.linewidths[0]
            C = _contour.Cntr(x, y, z.filled(), ma.getmaskorNone(z))
            lowers = self._levels[:-1]
            uppers = self._levels[1:]
            for level, level_upper in zip(lowers, uppers):
                nlist = C.trace(level, level_upper, points = 0,
                        nchunk = self.nchunk)
                col = PolyCollection(nlist,
                                     linewidths = (self.linewidths,),
                                     antialiaseds = (self.antialiased,),
                                     edgecolors= 'None')
                self.ax.add_collection(col)
                self.collections.append(col)

        else:
            tlinewidths = self._process_linewidths()
            self.tlinewidths = tlinewidths
            C = _contour.Cntr(x, y, z.filled(), ma.getmaskorNone(z))
            for level, width in zip(self.levels, tlinewidths):
                nlist = C.trace(level, points = 0)
                col = LineCollection(nlist,
                                     linewidths = width)

                if level < 0.0 and self.monochrome:
                    col.set_linestyle((0, rcParams['contour.negative_linestyle']))
                col.set_label(str(level))         # only for self-documentation
                self.ax.add_collection(col)
                self.collections.append(col)
        self.changed() # set the colors
        x0 = ma.minimum(x)
        x1 = ma.maximum(x)
        y0 = ma.minimum(y)
        y1 = ma.maximum(y)
        self.ax.update_datalim([(x0,y0), (x1,y1)])
        self.ax.set_xlim((x0, x1))
        self.ax.set_ylim((y0, y1))

    def changed(self):
        tcolors = [ (tuple(rgba),) for rgba in
                                self.to_rgba(self.cvalues, alpha=self.alpha)]
        self.tcolors = tcolors
        for color, collection in zip(tcolors, self.collections):
            collection.set_color(color)
        for label, cv in zip(self.cl, self.cl_cvalues):
            label.set_color(self.label_mappable.to_rgba(cv))
        # add label colors
        ScalarMappable.changed(self)


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
        zmax = self.zmax
        zmin = self.zmin
        zmargin = (zmax - zmin) * 0.001 # so z < (zmax + zmargin)
        zmax = zmax + zmargin
        intv = Interval(Value(zmin), Value(zmax))
        if self.locator is None:
            self.locator = MaxNLocator(N+1)
        self.locator.set_view_interval(intv)
        self.locator.set_data_interval(intv)
        lev = self.locator()
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
        if len(shape(z)) != 2:
            raise TypeError("Input must be a 2D array.")
        else:
            Ny, Nx = shape(z)
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return meshgrid(arange(Nx), arange(Ny))
            else:
                x0,x1,y0,y1 = self.extent
                x = linspace(x0, x1, Nx)
                y = linspace(y0, y1, Ny)
                return meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0,x1,y0,y1 = (0, Nx, 0, Ny)
        else:
            x0,x1,y0,y1 = self.extent
        dx = float(x1 - x0)/Nx
        dy = float(y1 - y0)/Ny
        x = x0 + (arange(Nx) + 0.5) * dx
        y = y0 + (arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return meshgrid(x,y)

    def _check_xyz(self, args):
        '''
        For functions like contour, check that the dimensions
        of the input arrays match; if x and y are 1D, convert
        them to 2D using meshgrid.

        Possible change: I think we should make and use an ArgumentError
        Exception class (here and elsewhere).

        Add checking for everything being the same numerix flavor?
        '''
        x,y,z = args
        if len(shape(z)) != 2:
            raise TypeError("Input z must be a 2D array.")
        else: Ny, Nx = shape(z)
        if shape(x) == shape(z) and shape(y) == shape(z):
            return x,y,z
        if len(shape(x)) != 1 or len(shape(y)) != 1:
            raise TypeError("Inputs x and y must be 1D or 2D.")
        nx, = shape(x)
        ny, = shape(y)
        if nx != Nx or ny != Ny:
            raise TypeError("Length of x must be number of columns in z,\n" +
                            "and length of y must be number of rows.")
        x,y = meshgrid(x,y)
        return x,y,z



    def _contour_args(self, *args):
        if self.filled: fn = 'contourf'
        else:           fn = 'contour'
        Nargs = len(args)
        if Nargs <= 2:
            z = args[0]
            x, y = self._initialize_x_y(z)
        elif Nargs <=4:
            x,y,z = self._check_xyz(args[:3])
        else:
            raise TypeError("Too many arguments to %s; see help(%s)" % (fn,fn))
        z = ma.asarray(z)  # Convert to native masked array format if necessary.
        self.zmax = ma.maximum(z)
        self.zmin = ma.minimum(z)
        self._auto = False
        if self.levels is None:
            if Nargs == 1 or Nargs == 3:
                lev = self._autolev(z, 7)
            else:   # 2 or 4 args
                level_arg = args[-1]
                if type(level_arg) == int:
                    lev = self._autolev(z, level_arg)
                elif iterable(level_arg) and len(shape(level_arg)) == 1:
                    lev = array([float(fl) for fl in level_arg])
                else:
                    raise TypeError("Last %s arg must give levels; see help(%s)" % (fn,fn))
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
        self._levels = asarray(self._levels)
        self.vmin = amin(self.levels)  # alternative would be self.layers
        self.vmax = amax(self.levels)
        if self.extend in ('both', 'min') or self.clip_ends:
            self.vmin = 2 * self.levels[0] - self.levels[1]
        if self.extend in ('both', 'max') or self.clip_ends:
            self.vmax = 2 * self.levels[-1] - self.levels[-2]
        self.layers = self._levels # contour: a line is a thin layer
        if self.filled:
            self.layers = 0.5 * (self._levels[:-1] + self._levels[1:])
            if self.extend in ('both', 'min') or self.clip_ends:
                self.layers[0] = 0.5 * (self.vmin + self._levels[1])
            if self.extend in ('both', 'max') or self.clip_ends:
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
            self.set_norm(NoNorm())
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
            tlinewidths = [(rcParams['lines.linewidth'],)] *Nlev
        else:
            if iterable(linewidths) and len(linewidths) < Nlev:
                linewidths = list(linewidths) * int(ceil(Nlev/len(linewidths)))
            elif not iterable(linewidths) and type(linewidths) in [int, float]:
                linewidths = [linewidths] * Nlev
            tlinewidths = [(w,) for w in linewidths]
        return tlinewidths

    def get_alpha(self):
        '''For compatibility with artists, return self.alpha'''
        return self.alpha

    def set_alpha(self, alpha):
        '''For compatibility with artists, set self.alpha'''
        self.alpha = alpha
        self.changed()

    contour_doc = """
        contour and contourf draw contour lines and filled contours,
        respectively.  Except as noted, function signatures and return
        values are the same for both versions.

        contourf differs from the Matlab (TM) version in that it does not
            draw the polygon edges, because the contouring engine yields
            simply connected regions with branch cuts.  To draw the edges,
            add line contours with calls to contour.


        Function signatures

        contour(Z) - make a contour plot of an array Z. The level
                 values are chosen automatically.

        contour(X,Y,Z) - X,Y specify the (x,y) coordinates of the surface

        contour(Z,N) and contour(X,Y,Z,N) - contour N automatically-chosen
                 levels.

        contour(Z,V) and contour(X,Y,Z,V) - draw len(V) contour lines,
                 at the values specified in sequence V

        contourf(..., V) - fill the (len(V)-1) regions between the
                 values in V

        contour(Z, **kwargs) - Use keyword args to control colors, linewidth,
                    origin, cmap ... see below

        X, Y, and Z must be arrays with the same dimensions.
        Z may be a masked array, but filled contouring may not handle
                   internal masked regions correctly.

        C = contour(...) returns a ContourSet object.


        Optional keyword args are shown with their defaults below (you must
        use kwargs for these):

            * colors = None; or one of the following:
              - a tuple of matplotlib color args (string, float, rgb, etc),
              different levels will be plotted in different colors in the order
              specified

              -  one string color, e.g. colors = 'r' or colors = 'red', all levels
              will be plotted in this color

              - if colors == None, the colormap specified by cmap will be used

            * alpha=1.0 : the alpha blending value

            * cmap = None: a cm Colormap instance from matplotlib.cm.
              - if cmap == None and colors == None, a default Colormap is used.

            * norm = None: a matplotlib.colors.Normalize instance for
              scaling data values to colors.
              - if norm == None, and colors == None, the default
                linear scaling is used.

            * origin = None: 'upper'|'lower'|'image'|None.
              If 'image', the rc value for image.origin will be used.
              If None (default), the first value of Z will correspond
              to the lower left corner, location (0,0).
              This keyword is active only if contourf is called with
              one or two arguments, that is, without explicitly
              specifying X and Y.

            * extent = None: (x0,x1,y0,y1); also active only if X and Y
              are not specified.  If origin is not None, then extent is
              interpreted as in imshow: it gives the outer pixel boundaries.
              In this case, the position of Z[0,0] is the center of the
              pixel, not a corner.
              If origin is None, then (x0,y0) is the position of Z[0,0],
              and (x1,y1) is the position of Z[-1,-1].

            * locator = None: an instance of a ticker.Locator subclass;
              default is MaxNLocator.  It is used to determine the
              contour levels if they are not given explicitly via the
              V argument.

            ***** New: *****
            * extend = 'neither', 'both', 'min', 'max'
              Unless this is 'neither' (default), contour levels are
              automatically added to one or both ends of the range so that
              all data are included.  These added ranges are then
              mapped to the special colormap values which default to
              the ends of the colormap range, but can be set via
              Colormap.set_under() and Colormap.set_over() methods.
              To replace clip_ends=True and V = [-100, 2, 1, 0, 1, 2, 100],
              use extend='both' and V = [2, 1, 0, 1, 2].
            ****************

            contour only:
            * linewidths = None: or one of these:
              - a number - all levels will be plotted with this linewidth,
                e.g. linewidths = 0.6

              - a tuple of numbers, e.g. linewidths = (0.4, 0.8, 1.2) different
                levels will be plotted with different linewidths in the order
                specified

              - if linewidths == None, the default width in lines.linewidth in
                matplotlibrc is used

            contourf only:
            ***** Obsolete: ****
            * clip_ends = True
              If False, the limits for color scaling are set to the
              minimum and maximum contour levels.
              True (default) clips the scaling limits.  Example:
              if the contour boundaries are V = [-100, 2, 1, 0, 1, 2, 100],
              then the scaling limits will be [-100, 100] if clip_ends
              is False, and [-3, 3] if clip_ends is True.
            * linewidths = None or a number; default of 0.05 works for
              Postscript; a value of about 0.5 seems better for Agg.
            * antialiased = True (default) or False; if False, there is
              no need to increase the linewidths for Agg, but True gives
              nicer color boundaries.  If antialiased is True and linewidths
              is too small, then there may be light-colored lines at the
              color boundaries caused by the antialiasing.
            * nchunk = 0 (default) for no subdivision of the domain;
              specify a positive integer to divide the domain into
              subdomains of roughly nchunk by nchunk points. This may
              never actually be advantageous, so this option may be
              removed.  Chunking introduces artifacts at the chunk
              boundaries unless antialiased = False, or linewidths is
              set to a large enough value for the particular renderer and
              resolution.
        """


