"""
These are helper functions and classes to support contour plotting and
labelling for the axes class
"""
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
from colors import colorConverter, normalize, Colormap, LinearSegmentedColormap
from collections import RegularPolyCollection, PolyCollection, LineCollection
from font_manager import FontProperties
from numerix.mlab import flipud, amin, amax
from text import Text



class ContourMappable(ScalarMappable):
    """
    a class to allow contours to respond properly to change in cmaps, etc
    """
    def __init__(self, levels, collections, norm=None, cmap=None, labeld=None):
        """
        See comment on labeld in the ContourLabeler class
        
        """
        ScalarMappable.__init__(self, norm, cmap)
        self.levels = levels
        self.collections = collections
        if labeld is None: labeld = {}
        self.labeld = labeld

    def changed(self):
        colors = [ (tuple(rgba),) for rgba in self.to_rgba(self.levels)]
        contourNum = 0
        for color, collection in zip(colors, self.collections):
            collection.set_color(color)
            Ncolor = len(color) # collections could have more than 1 in principle
            for segNum, segment in enumerate(collection._segments):
                key = contourNum, segNum
                t = self.labeld.get(key)
                if t is not None: t.set_color(color[segNum%Ncolor])
            contourNum += 1
            
        ScalarMappable.changed(self)

class ContourLabeler:
    def __init__(self, ax):
        self.ax = ax
        
    def clabel(self, *args, **kwargs):
        """
        CLABEL(*args, **kwargs)

        Function signatures

        CLABEL(C) - plots contour labels,
                    C is the output of contour or a list of contours

        CLABEL(C,V) - creates labels only for those contours, given in
                      a list V

        CLABEL(C, **kwargs) - keyword args are explained below:



        * fontsize = None: as described in http://matplotlib.sf.net/fonts.html

        * colors = None:

           - a tuple of matplotlib color args (string, float, rgb, etc),
             different labels will be plotted in different colors in the order
             specified

           - one string color, e.g. colors = 'r' or colors = 'red', all labels
             will be plotted in this color

           - if colors == None, the color of each label matches the color
             of the corresponding contour

        * inline = 0: controls whether the underlying contour is removed
                     (inline = 1) or not

        * fmt = '%1.3f': a format string for the label

        """
        # todo, factor this out to a separate class and don't use hidden coll attrs

        if not self.ax.ishold(): self.ax.cla()

        fontsize = kwargs.get('fontsize', None)
        inline = kwargs.get('inline', 0)
        fmt = kwargs.get('fmt', '%1.3f')
        colors = kwargs.get('colors', None)



        if len(args) == 1:
            contours = args[0]
            levels = [con._label for con in contours]
        elif len(args) == 2:
            contours = args[0]
            levels = args[1]
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
            colors = [c._colors[0] for c in contours]
        else:
            colors = colors * len(contours)

        if inline not in [0,1]:
            raise TypeError("inline must be 0 or 1")


        self.cl = []
        self.cl_xy = []

        # we have a list of contours and each contour has a list of
        # segments.  We want changes in the contour color to be
        # reflected in changes in the label color.  This is a good use
        # for traits observers, but in the interim, until traits are
        # utilized, we'll create a dict mapping i,j to text instances.
        # i is the contour level index, j is the sement index
        self.labeld = {}
        if inline == 1:
            toremove, toadd = self.inline_labels(levels, contours, colors, fslist, fmt)
            for r in toremove:
                self.ax.collections.remove(r)
            for a in toadd:
                self.ax.add_collection(a)
        else:
            self.labels(levels, contours, colors, fslist)

        for label in self.cl:
            self.ax.add_artist(label)


        if hasattr(contours, 'mappable'):
            old = getattr(contours, 'mappable')
            mappable = ContourMappable(old.get_array(), toadd, cmap=old.cmap, labeld=self.labeld)            
            mappable.set_array(old.get_array())
            mappable.autoscale()
        else:
            mappable = None

        ret =  silent_list('Text', self.cl)
        ret.mappable = mappable
        return ret



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
        if (xmax - xmin) > 3* lw or (ymax - ymin) > 3 * lw:
            return 1
        else:
            return 0

    def too_close(self, x,y):
        "if there's a label already nearby, find a better place"
        if self.cl_xy != []:
            dist = [sqrt((x-loc[0]) ** 2 + (y-loc[1]) ** 2) for loc in self.cl_xy]
            for d in dist:
                if d < 150:
                    return 1
                else: return 0
        else: return 0

    def get_label_coords(self, distances, XX, YY, ysize):
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
            if self.too_close(x,y):
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


    def set_label_props(self, label,text, color):
        "set the label properties - color, fontsize, text"
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self.fp)


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

        inds=nonzero(((xx < x+xlabel) & (xx > x-xlabel)) & ((yy < y+ylabel) & (yy > y-ylabel)))

        if len(inds) != 0:
            lc1=linecontour[:inds[0]]
            lc2 = linecontour[inds[-1]+1:]
        else:
            lc1=linecontour[:ind]
            lc2 = linecontour[ind+1:]
        epsilon=.000005

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
        s = (reshape(yfirst, (xsize,1))-YY)*(reshape(xlast,(xsize,1))-reshape(xfirst,(xsize,1)))-(reshape(xfirst,(xsize,1))-XX)*(reshape(ylast,(xsize,1))-reshape(yfirst,(xsize,1)))
        L=sqrt((xlast-xfirst)**2+(ylast-yfirst)**2)
        dist = add.reduce(([(abs(s)[i]/L[i]) for i in range(xsize)]),-1)
        ind = argmin(dist)
        x, y = XX[ind][int(ysize/2)], YY[ind][int(ysize/2)]
        x,y,ind = self.get_label_coords(dist, XX, YY, ysize)
        angle = arctan2(ylast - yfirst, xlast - xfirst)
        rotation = angle[ind]*180/pi
        if rotation > 90:
            rotation = rotation -180
        if rotation < -90:
            rotation = 180 + rotation

        dind = list(linecontour).index((x,y))

        return x,y, rotation, dind

    def inline_labels(self, levels, contours, colors, fslist, fmt):
        toremove = []
        toadd = []
        trans = self.ax.transData
        contourNum = 0
        for lev, con, color, fsize in zip(levels, contours, colors, fslist):
            col = []
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
                    t = Text(dx, dy, rotation = rotation, horizontalalignment='center', verticalalignment='center')
                    self.labeld[key] = t
                    text = self.get_text(lev,fmt)
                    self.set_label_props(t, text, color)
                    self.cl.append(t)

                    new  =  self.break_linecontour(linecontour, rotation, lw, ind)
                    for c in new: col.append(c)
                else:
                    col.append(linecontour)
            toremove.append(con)
            toadd.append(LineCollection(col, colors=con._colors, linewidths = con._lw))
            contourNum += 1

        return toremove, toadd

    def labels(self, levels, contours, colors, fslist):
        trans = self.ax.transData
        for lev, con, color, fsize in zip(levels, contours, colors, fslist):
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
                    t = Text(dx, dy, rotation = rotation, horizontalalignment='center', verticalalignment='center')
                    text = self.get_text(lev, fmt)
                    self.set_label_props(t, text, color)
                    self.cl.append(t)
                else:
                    pass


    

class ContourSupport:

    def __init__(self, ax):
        """
        Provide a reference to ax
        """
        self.ax = ax
        self.labeler = ContourLabeler(ax)
        
    def _autolev(self, z, N, filled, badmask):
        '''
        Select contour levels to span the data.

        We need one more level for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        two levels.  These are taken as the lower boundaries of
        the regions.
        '''
        rz = ma.masked_array(z, badmask)
        zmax = ma.maximum(rz)     # was: zmax = amax(rz)
        zmin = ma.minimum(rz)
        if filled:
            lev = linspace(zmin, zmax, N+2)[:-1]
        else:
            lev = linspace(zmin, zmax, N+2)[1:-1]
        return lev

    def _initialize_x_y(self, z, origin):
        '''
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i,j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        '''
        if len(shape(z)) != 2:
            raise TypeError("Input must be a 2D array.")
        else: imax, jmax = shape(z)
        y = arange(imax)
        if origin is None:  shift = 0.0
        else:               shift = 0.5
        if origin == 'upper':
            y = y[::-1]
        return meshgrid(arange(jmax)+ shift, y + shift)

    def _check_xyz(self, args):
        '''
        For functions like contour, check that the dimensions
        of the input arrays match; if x and y are 1D, convert
        them to 2D using meshgrid.

        Possible change: I think we should make and use an ArgumentError
        Exception class (here and elsewhere).
        '''
        x,y,z = args
        if len(shape(z)) != 2:
            raise TypeError("Input z must be a 2D array.")
        else: imax, jmax = shape(z)
        if shape(x) == shape(z) and shape(y) == shape(z):
            return x,y,z
        if len(shape(x)) != 1 or len(shape(y)) != 1:
            raise TypeError("Inputs x and y must be 1D or 2D.")
        nx, = shape(x)
        ny, = shape(y)
        if nx != jmax or ny != imax:
            raise TypeError("Length of x must be number of columns in z,\n" +
                            "and length of y must be number of rows.")
        x,y = meshgrid(x,y)
        return x,y,z



    def _contour_args(self, filled, badmask, origin, *args):
        if filled: fn = 'contourf'
        else:      fn = 'contour'
        Nargs = len(args)
        if Nargs <= 2:
            z = args[0]
            x, y = self._initialize_x_y(z, origin)
        elif Nargs <=4:
            x,y,z = self._check_xyz(args[:3])
        else:
            raise TypeError("Too many arguments to %s; see help(%s)" % (fn,fn))
        z = asarray(z)  # Convert to native array format if necessary.
        if Nargs == 1 or Nargs == 3:
            lev = self._autolev(z, 7, filled, badmask)
        else:
            level_arg = args[-1]
            if type(level_arg) == int:
                lev = self._autolev(z, level_arg, filled, badmask)
            elif iterable(level_arg) and len(shape(level_arg)) == 1:
                lev = array([float(fl) for fl in level_arg])
            else:
                raise TypeError("Last %s arg must give levels; see help(%s)" % (fn,fn))
        self.ax.set_xlim((min(ravel(x)), max(ravel(x))))
        self.ax.set_ylim((min(ravel(y)), max(ravel(y))))
        return (x, y, z, lev)


    def _initialize_reg_tri(self, z, badmask):
        '''
        Initialize two arrays used by the low-level contour
        algorithm.  This is temporary code; most of the reg
        initialization should be done in c.

        For each masked point, we need to mark as missing
        the four regions with that point as a corner.
        '''
        imax, jmax = shape(z)
        nreg = jmax*(imax+1)+1
        reg = ones((1, nreg), typecode = 'i')
        reg[0,:jmax+1]=0
        reg[0,-jmax:]=0
        for j in range(0, nreg, jmax):
            reg[0,j]=0
        if badmask is not None:
            for i in range(imax):
                for j in range(jmax):
                    if badmask[i,j]:
                        ii = i*jmax+j
                        if ii < nreg:
                            reg[0,ii] = 0
                        ii += 1
                        if ii < nreg:
                            reg[0,ii] = 0
                        ii += jmax
                        if ii < nreg:
                            reg[0,ii] = 0
                        ii -= 1
                        if ii < nreg:
                            reg[0,ii] = 0

        triangle = zeros((imax,jmax), typecode='s')

        return reg, triangle

    def _process_colors(self, z, colors, alpha, lev, cmap):
        """
        Color argument processing for contouring.

        Note that we base the color mapping on the contour levels,
        not on the actual range of the Z values.  This means we
        don't have to worry about bad values in Z, and we always have
        the full dynamic range available for the selected levels.

        The input argument Z is not actually being used now; if
        using the levels for autoscaling instead of Z works well,
        then it will be removed.
        """
        Nlev = len(lev)
        collections = []
        if colors is not None:

            if is_string_like(colors):
                colors = [colors] * Nlev
            elif iterable(colors) and len(colors) < Nlev:
                colors = list(colors) * Nlev
            else:
                try: gray = float(colors)
                except TypeError: pass
                else:  colors = [gray] * Nlev

            tcolors = [(colorConverter.to_rgba(c, alpha),) for c in colors]
            mappable = None
        else:
            mappable = ContourMappable(lev, collections, cmap=cmap)
            #mappable.set_array(z)
            mappable.set_array(lev)
            mappable.autoscale()
            tcolors = [ (tuple(rgba),) for rgba in mappable.to_rgba(lev)]
        return tcolors, mappable, collections

    def contour(self, *args, **kwargs):
        """
        contour(self, *args, **kwargs)

        Function signatures

        contour(Z) - make a contour plot of an array Z. The level
                 values are chosen automatically.

        contour(X,Y,Z) - X,Y specify the (x,y) coordinates of the surface

        contour(Z,N) and contour(X,Y,Z,N) - draw N contour lines overriding
                         the automatic value

        contour(Z,V) and contour(X,Y,Z,V) - draw len(V) contour lines,
                       at the values specified in V (array, list, tuple)

        contour(Z, **kwargs) - Use keyword args to control colors, linewidth,
                    origin, cmap ... see below

        [L,C] = contour(...) returns a list of levels and a silent_list of LineCollections

        Optional keywork args are shown with their defaults below (you must
        use kwargs for these):

            * colors = None: one of these:
              - a tuple of matplotlib color args (string, float, rgb, etc),
              different levels will be plotted in different colors in the order
              specified

              -  one string color, e.g. colors = 'r' or colors = 'red', all levels
              will be plotted in this color

              - if colors == None, the default colormap will be used

            * alpha=1.0 : the alpha blending value

            * cmap = None: a cm Colormap instance from matplotlib.cm.

            * origin = None: 'upper'|'lower'|'image'|None.
              If 'image', the rc value for image.origin will be used.
              If None (default), the first value of Z will correspond
              to the lower left corner, location (0,0).
              This keyword is active only if contourf is called with
              one or two arguments, that is, without explicitly
              specifying X and Y.

            * badmask = None: array with dimensions of Z, and with values
              of zero at locations corresponding to valid data, and one
              at locations where the value of Z should be ignored.
              This is experimental.  It presently works for edge regions
              for line and filled contours, but for interior regions it
              works correctly only for line contours.  The badmask kwarg
              may go away in the future, to be replaced by the use of
              NaN value in Z and/or the use of a masked array in Z.

            * linewidths = None: or one of these:
              - a number - all levels will be plotted with this linewidth,
                e.g. linewidths = 0.6

              - a tuple of numbers, e.g. linewidths = (0.4, 0.8, 1.2) different
                levels will be plotted with different linewidths in the order
                specified

              - if linewidths == None, the default width in lines.linewidth in
                .matplotlibrc is used

            * fmt = '1.3f': a format string for adding a label to each collection.
              Useful for auto-legending.

        """

        alpha = kwargs.get('alpha', 1.0)
        linewidths = kwargs.get('linewidths', None)
        fmt = kwargs.get('fmt', '%1.3f')
        origin = kwargs.get('origin', None)
        cmap = kwargs.get('cmap', None)
        colors = kwargs.get('colors', None)
        badmask = kwargs.get('badmask', None)

        if cmap is not None: assert(isinstance(cmap, Colormap))
        if origin is not None: assert(origin in ['lower', 'upper', 'image'])

        if colors is not None and cmap is not None:
            raise RuntimeError('Either colors or cmap must be None')
        if origin == 'image': origin = rcParams['image.origin']


        x, y, z, lev = self._contour_args(False, badmask, origin, *args)

        # Manipulate the plot *after* checking the input arguments.
        if not self.ax.ishold(): self.ax.cla()

        Nlev = len(lev)


        reg, triangle = self._initialize_reg_tri(z, badmask)

        tcolors, mappable, collections = self._process_colors(z, colors,
                                                              alpha,
                                                              lev, cmap)

        if linewidths == None:
            tlinewidths = [rcParams['lines.linewidth']] *Nlev
        else:
            if iterable(linewidths) and len(linewidths) < Nlev:
                linewidths = list(linewidths) * int(ceil(Nlev/len(linewidths)))
            elif not iterable(linewidths) and type(linewidths) in [int, float]:
                linewidths = [linewidths] * Nlev
            tlinewidths = [(w,) for w in linewidths]

        region = 0
        for level, color, width in zip(lev, tcolors, tlinewidths):
            ntotal, nparts  = _contour.GcInit1(x, y, reg, triangle,
                                               region, z, level)
            np = zeros((nparts,), typecode='l')
            xp = zeros((ntotal, ), Float64)
            yp = zeros((ntotal,), Float64)
            nlist = _contour.GcTrace(np, xp, yp)
            col = LineCollection(nlist)
            col.set_color(color)
            col.set_linewidth(width)

            if level < 0:
                col.set_linestyle((0, (6.,6.)),)

            col.set_label(fmt%level)
            self.ax.add_collection(col)
            collections.append(col)

        collections = silent_list('LineCollection', collections)
        collections.mappable = mappable
        return lev, collections



    def contourf(self, *args, **kwargs):
        """
        contourf(self, *args, **kwargs)

        Function signatures

        contourf(Z) - make a filled contour plot of an array Z. The level
                 values are chosen automatically.

        contourf(X,Y,Z) - X,Y specify the (x,y) coordinates of the surface

        contourf(Z,N) and contourf(X,Y,Z,N) - make a filled contour plot
                 corresponding to N contour levels

        contourf(Z,V) and contourf(X,Y,Z,V) - fill len(V) regions,
                 between the levels specified in sequence V, and a final region
                 for values of Z greater than the last element in V

        contourf(Z, **kwargs) - Use keyword args to control colors,
                    origin, cmap ... see below

        [L,C] = contourf(...) returns a list of levels and a silent_list
             of PolyCollections

        Optional keywork args are shown with their defaults below (you must
        use kwargs for these):

            * colors = None: one of these:
              - a tuple of matplotlib color args (string, float, rgb, etc),
              different levels will be plotted in different colors in the order
              specified

              -  one string color, e.g. colors = 'r' or colors = 'red', all levels
              will be plotted in this color

              - if colors == None, the default colormap will be used

            * alpha=1.0 : the alpha blending value

            * cmap = None: a cm Colormap instance from matplotlib.cm.

            * origin = None: 'upper'|'lower'|'image'|None.
              If 'image', the rc value for image.origin will be used.
              If None (default), the first value of Z will correspond
              to the lower left corner, location (0,0).
              This keyword is active only if contourf is called with
              one or two arguments, that is, without explicitly
              specifying X and Y.

            * badmask = None: array with dimensions of Z, and with values
              of zero at locations corresponding to valid data, and one
              at locations where the value of Z should be ignored.
              This is experimental.  It presently works for edge regions
              for line and filled contours, but for interior regions it
              works correctly only for line contours.  The badmask kwarg
              may go away in the future, to be replaced by the use of
              NaN value in Z and/or the use of a masked array in Z.

            reg is a 1D region number array with of imax*(jmax+1)+1 size
            The values of reg should be positive region numbers, and zero fro
            zones wich do not exist.

            triangle - triangulation array - must be the same shape as reg

            contourf differs from the Matlab (TM) version in that it does not
                draw the polygon edges (because the contouring engine yields
                simply connected regions with branch cuts.)  To draw the edges,
                add line contours with calls to contour.

        """

        alpha = kwargs.get('alpha', 1.0)
        origin = kwargs.get('origin', None)
        cmap = kwargs.get('cmap', None)
        colors = kwargs.get('colors', None)
        badmask = kwargs.get('badmask', None)

        if cmap is not None: assert(isinstance(cmap, Colormap))
        if origin is not None: assert(origin in ['lower', 'upper', 'image'])

        if colors is not None and cmap is not None:
            raise RuntimeError('Either colors or cmap must be None')
        if origin == 'image': origin = rcParams['image.origin']

        x, y, z, lev = self._contour_args(True, badmask, origin, *args)
        # Manipulate the plot *after* checking the input arguments.
        if not self.ax.ishold(): self.ax.cla()

        Nlev = len(lev)


        reg, triangle = self._initialize_reg_tri(z, badmask)

        tcolors, mappable, collections = self._process_colors(z, colors,
                                                               alpha,
                                                               lev, cmap)

        region = 0
        lev_upper = list(lev[1:])
        lev_upper.append(1e38)
        for level, level_upper, color in zip(lev, lev_upper, tcolors):
            levs = (level, level_upper)
            ntotal, nparts  = _contour.GcInit2(x, y, reg, triangle,
                                               region, z, levs, 30)
            np = zeros((nparts,), typecode='l')
            xp = zeros((ntotal, ), Float64)
            yp = zeros((ntotal,), Float64)
            nlist = _contour.GcTrace(np, xp, yp)
            col = PolyCollection(nlist,
                                         linewidths=(1,))
                  # linewidths = 1 is necessary to avoid artifacts
                  # in rendering the region boundaries.
            col.set_color(color) # sets both facecolor and edgecolor
            #col.set_edgecolor(color)

            self.ax.add_collection(col)
            collections.append(col)

        collections = silent_list('PolyCollection', collections)
        collections.mappable = mappable
        return lev, collections


