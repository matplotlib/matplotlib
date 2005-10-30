"""
These are  classes to support contour plotting and
labelling for the axes class
"""
from __future__ import division
from matplotlib import rcParams
import numerix.ma as ma

from numerix import absolute, arange, array, asarray, ones, divide,\
     transpose, log, log10, Float, Float32, ravel, zeros, Int16,\
     Int32, Int, Float64, ceil, indices, shape, which, where, sqrt,\
     asum, resize, reshape, add, argmin, arctan2, pi, argsort, sin,\
     cos, nonzero

from mlab import linspace, meshgrid
import _contour
from cm import ScalarMappable
from cbook import iterable, is_string_like, flatten, enumerate, \
     allequal, dict_delall, strip_math, popd, popall, silent_list
from colors import colorConverter, normalize, Colormap
from collections import  PolyCollection, LineCollection
from font_manager import FontProperties
from numerix.mlab import amin, amax
from text import Text


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
        elif len(args) == 1:
            levels = [lev for lev in args[0] if lev in self.levels]
            if len(levels) < len(args[0]):
                msg = "Specified levels " + str(levels)
                msg += "\n don't match available levels "
                msg += str(self.levels)
                raise ValueError(msg)
        else:
            raise TypeError("Illegal arguments to clabel, see help(clabel)")

        self.fp = FontProperties()
        if fontsize == None:
            font_size = int(self.fp.get_size_in_points())
        else:
            if type(fontsize) not in [int, float, str]:
                raise TypeError("Font size must be an integer number.")
            else:
                if type(fontsize) == str:
                    font_size = int(self.fp.get_size_in_points())

                else:
                    self.fp.set_size(fontsize)
                    font_size = fontsize
        fslist = [font_size] * len(levels)

        if colors == None:
            self.labelcolors = [c[0] for c in self.tcolors]
        else:
            self.labelcolors = colors * len(self.collections)

        self.cl = []
        self.cl_xy = []

        # we have a list of contours and each contour has a list of
        # segments.  We want changes in the contour color to be
        # reflected in changes in the label color.  This is a good use
        # for traits observers, but in the interim, until traits are
        # utilized, we'll create a dict mapping i,j to text instances.
        # i is the contour level index, j is the segment index
        if inline:
            self.inline_labels(levels, fslist)
        else:
            self.labels(levels, fslist)

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
        xx= array(slc)[:,0].copy()
        yy=array(slc)[:,1].copy()

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

        if rot > 0:
            if len(lc1) > 0 and (lc1[-1][0] <= new_x1d) and (lc1[-1][1] <= new_y1d):
                lc1.append((new_x1d, new_y1d))

            if len(lc2) > 0 and (lc2[0][0] >= new_x2d) and (lc2[0][1] >= new_y2d):
                lc2.insert(0, (new_x2d, new_y2d))
        else:
            if len(lc1) > 0 and ((lc1[-1][0] <= new_x1d) and (lc1[-1][1] >= new_y1d)):
                lc1.append((new_x1d, new_y1d))

            if len(lc2) > 0 and ((lc2[0][0] >= new_x2d) and (lc2[0][1] <= new_y2d)):
                lc2.insert(0, (new_x2d, new_y2d))

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

        XX = resize(array(linecontour)[:,0],(xsize, ysize))
        YY = resize(array(linecontour)[:,1],(xsize,ysize))

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
        angle = arctan2(ylast - yfirst, xlast - xfirst)
        rotation = angle[ind]*180/pi
        if rotation > 90:
            rotation = rotation -180
        if rotation < -90:
            rotation = 180 + rotation

        dind = list(linecontour).index((x,y))

        return x,y, rotation, dind

    def inline_labels(self, levels, fslist):
        trans = self.ax.transData
        contourNum = 0
        colors = self.labelcolors
        contours = self.collections
        fmt = self.fmt
        for lev, con, color, fsize in zip(self.levels, contours, colors, fslist):
            if lev not in levels:
                continue
            toremove = []
            toadd = []
            lw = self.get_label_width(lev, fmt, fsize)
            for segNum, linecontour in enumerate(con._segments):
                key = contourNum, segNum
                # for closed contours add one more point to
                # avoid division by zero
                if linecontour[0] == linecontour[-1]:
                    linecontour.append(linecontour[1])
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
                    self.labeld[key] = t
                    text = self.get_text(lev,fmt)
                    self.set_label_props(t, text, color)
                    self.cl.append(t)
                    new  =  self.break_linecontour(linecontour, rotation,
                                                   lw, ind)

                    for c in new: toadd.append(c)
                    toremove.append(linecontour)
            for c in toremove:
                con._segments.remove(c)
            for c in toadd: con._segments.append(c)

            contourNum += 1


    def labels(self, levels, fslist):
        trans = self.ax.transData
        colors = self.labelcolors
        contours = self.collections
        fmt = self.fmt
        for lev, con, color, fsize in zip(self.levels, contours, colors, fslist):
            if lev not in levels:
                continue
            lw = self.get_label_width(lev, fmt, fsize)
            for linecontour in con._segments:
                # for closed contours add one more point
                if linecontour[0] == linecontour[-1]:
                    linecontour.append(linecontour[1])
                # transfer all data points to screen coordinates
                slc = trans.seq_xy_tups(linecontour)
                if self.print_label(slc,lw):
                    x,y, rotation, ind  = self.locate_label(slc, lw)
                    # transfer the location of the label back into
                    # data coordinates
                    dx,dy = trans.inverse_xy_tup((x,y))
                    t = Text(dx, dy, rotation = rotation,
                             horizontalalignment='center',
                             verticalalignment='center')
                    text = self.get_text(lev, fmt)
                    self.set_label_props(t, text, color)
                    self.cl.append(t)
                else:
                    pass




class ContourSet(ScalarMappable, ContourLabeler):
    """
    Create and store a set of contour lines or filled regions.

    User-callable method: clabel

    Useful attributes:
        ax - the axes object in which the contours are drawn
        collections - a silent_list of LineCollections or PolyCollections
        levels - contour levels
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
        self.filled = kwargs.get('filled', False)
        self.linewidths = kwargs.get('linewidths', None)
        #self.fmt = kwargs.get('format', '%1.3f')

        self.alpha = kwargs.get('alpha', 1.0)
        self.origin = kwargs.get('origin', None)
        self.extent = kwargs.get('extent', None)
        cmap = kwargs.get('cmap', None)
        self.colors = kwargs.get('colors', None)
        self.clip_ends = kwargs.get('clip_ends', True)
        self.antialiased = kwargs.get('antialiased', True)
        self.nchunk = kwargs.get('nchunk', 0)


        if cmap is not None: assert(isinstance(cmap, Colormap))
        if self.origin is not None: assert(self.origin in
                                            ['lower', 'upper', 'image'])
        if self.extent is not None: assert(len(self.extent) == 4)
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image': self.origin = rcParams['image.origin']

        ScalarMappable.__init__(self, cmap = cmap) # sets self.cmap;
                                                   # default norm for now
        x, y, z = self._contour_args(*args)        # also sets self.levels



        self.labeld = {}
        # Note: _process_colors must follow initialization of self.collections.
        if self.filled:
            if self.linewidths is None:
                self.linewidths = 0.05 # Good default for Postscript.
            if iterable(self.linewidths):
                self.linewidths = self.linewidths[0]
            self.collections = silent_list('PolyCollection')
            self._process_colors()                     # sets self.tcolors
            C = _contour.Cntr(x, y, z.filled(), z.mask())
            lowers = self.levels[:-1]
            uppers = self.levels[1:]
            for level, level_upper, color in zip(lowers, uppers, self.tcolors):
                nlist = C.trace(level, level_upper, points = 1,
                        nchunk = self.nchunk)
                col = PolyCollection(nlist,
                                     linewidths = (self.linewidths,),
                                     antialiaseds = (self.antialiased,))
                col.set_color(color) # sets both facecolor and edgecolor
                self.ax.add_collection(col)
                self.collections.append(col)

        else:
            self.collections = silent_list('LineCollection')
            self._process_colors()                     # sets self.tcolors
            tlinewidths = self._process_linewidths()

            C = _contour.Cntr(x, y, z.filled(), z.mask())
            for level, color, width in zip(self.levels, self.tcolors, tlinewidths):
                nlist = C.trace(level, points = 1)
                col = LineCollection(nlist)
                col.set_color(color)
                col.set_linewidth(width)

                if level < 0.0 and self.monochrome:
                    col.set_linestyle((0, (6.,6.)),)
                    #print "setting dashed"
                col.set_label(str(level))         # only for self-documentation
                self.ax.add_collection(col)
                self.collections.append(col)

        ## check: seems like set_xlim should also be inside
        if not self.ax.ishold():
            self.ax.cla()
        self.ax.set_xlim((ma.minimum(x), ma.maximum(x)))
        self.ax.set_ylim((ma.minimum(y), ma.maximum(y)))



    def changed(self):
        colors = [ (tuple(rgba),) for rgba in self.to_rgba(self.levels)]
        contourNum = 0
        for color, collection in zip(colors, self.collections):
            collection.set_color(color)
            Ncolor = len(color) # collections could have more than 1 in principle

            segments = getattr(collection, '_segments', [])
            for segNum, segment in enumerate(segments):
                key = contourNum, segNum
                t = self.labeld.get(key)
                if t is not None: t.set_color(color[segNum%Ncolor])
            contourNum += 1

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
        zmax = ma.maximum(z)
        zmin = ma.minimum(z)
        zmargin = (zmax - zmin) * 0.001 # so z < (zmax + zmargin)
        if self.filled:
            lev = linspace(zmin, zmax + zmargin, N+2)
        else:
            lev = linspace(zmin, zmax + zmargin, N+2)[1:-1]
        return lev

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
        '''
        if len(shape(z)) != 2:
            raise TypeError("Input must be a 2D array.")
        else:
            Ny, Nx = shape(z)
        if self.origin is None:
            return meshgrid(arange(Nx), arange(Ny))

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
        return (x, y, z)

    def _process_colors(self):
        """
        Color argument processing for contouring.

        Note that we base the color mapping on the contour levels,
        not on the actual range of the Z values.  This means we
        don't have to worry about bad values in Z, and we always have
        the full dynamic range available for the selected levels.
        """
        self.layers = self.levels # contour: a line is a thin layer
        if self.filled:
            self.layers = 0.5 * (self.levels[:-1] + self.levels[1:])
        Nlayers = len(self.layers)
        self.monochrome = False
        if self.colors is not None:
            if is_string_like(self.colors):
                self.monochrome = True
                self.colors = [self.colors] * Nlayers
            elif iterable(self.colors) and len(self.colors) < Nlayers:
                self.colors = list(self.colors) * Nlayers
            else:
                try: gray = float(self.colors)
                except TypeError: pass
                else:  self.colors = [gray] * Nlayers

            tcolors = [(colorConverter.to_rgba(c, self.alpha),)
                        for c in self.colors]
        else:
            self.set_array(self.layers)
            if self.filled and Nlayers > 2 and self.clip_ends:
                vmin = 2 * self.levels[1] - self.levels[2]
                vmax = 2 * self.levels[-2] - self.levels[-3]
            else:
                vmin = amin(self.levels)
                vmax = amax(self.levels)
            self.set_clim(vmin, vmax)
            tcolors = [ (tuple(rgba),) for rgba in self.to_rgba(self.levels)]
        self.tcolors = tcolors

    def _process_linewidths(self):
        linewidths = self.linewidths
        Nlev = len(self.levels)
        if linewidths == None:
            tlinewidths = [rcParams['lines.linewidth']] *Nlev
        else:
            if iterable(linewidths) and len(linewidths) < Nlev:
                linewidths = list(linewidths) * int(ceil(Nlev/len(linewidths)))
            elif not iterable(linewidths) and type(linewidths) in [int, float]:
                linewidths = [linewidths] * Nlev
            tlinewidths = [(w,) for w in linewidths]
        return tlinewidths

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

            * origin = None: 'upper'|'lower'|'image'|None.
              If 'image', the rc value for image.origin will be used.
              If None (default), the first value of Z will correspond
              to the lower left corner, location (0,0).
              This keyword is active only if contourf is called with
              one or two arguments, that is, without explicitly
              specifying X and Y.

            * extent = None: (x0,x1,y0,y1); also active only if X and Y
              are not specified.

            contour only:
            * linewidths = None: or one of these:
              - a number - all levels will be plotted with this linewidth,
                e.g. linewidths = 0.6

              - a tuple of numbers, e.g. linewidths = (0.4, 0.8, 1.2) different
                levels will be plotted with different linewidths in the order
                specified

              - if linewidths == None, the default width in lines.linewidth in
                .matplotlibrc is used

            contourf only:
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


