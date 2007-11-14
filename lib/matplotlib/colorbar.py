'''
Colorbar toolkit with two classes and a function:

    ColorbarBase is the base class with full colorbar drawing functionality.
        It can be used as-is to make a colorbar for a given colormap;
        a mappable object (e.g., image) is not needed.

    Colorbar is the derived class for use with images or contour plots.

    make_axes is a function for resizing an axes and adding a second axes
        suitable for a colorbar

The Figure.colorbar() method uses make_axes and Colorbar; the pylab.colorbar()
function is a thin wrapper over Figure.colorbar().

'''

import numpy as npy
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook
import matplotlib.transforms as transforms
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.contour as contour

make_axes_kw_doc = '''
        fraction    = 0.15; fraction of original axes to use for colorbar
        pad         = 0.05 if vertical, 0.15 if horizontal; fraction
                              of original axes between colorbar and
                              new image axes
        shrink      = 1.0; fraction by which to shrink the colorbar
        aspect      = 20; ratio of long to short dimensions

'''

colormap_kw_doc = '''
        extend='neither', 'both', 'min', 'max'
                If not 'neither', make pointed end(s) for out-of-range
                values.  These are set for a given colormap using the
                colormap set_under and set_over methods.
        spacing='uniform', 'proportional'
                Uniform spacing gives each discrete color the same space;
                proportional makes the space proportional to the data interval.
        ticks=None, list of ticks, Locator object
                If None, ticks are determined automatically from the input.
        format=None, format string, Formatter object
                If none, the ScalarFormatter is used.
                If a format string is given, e.g. '%.3f', that is used.
                An alternative Formatter object may be given instead.
        drawedges=False, True
                If true, draw lines at color boundaries.

        The following will probably be useful only in the context of
        indexed colors (that is, when the mappable has norm=NoNorm()),
        or other unusual circumstances.

        boundaries=None or a sequence
        values=None or a sequence which must be of length 1 less than the
                sequence of boundaries.
                For each region delimited by adjacent entries in
                boundaries, the color mapped to the corresponding
                value in values will be used.

'''

colorbar_doc = '''
Add a colorbar to a plot.

Function signatures for the pyplot interface; all but the first are
also method signatures for the Figure.colorbar method:

    colorbar(**kwargs)
    colorbar(mappable, **kwargs)
    colorbar(mappable, cax=cax, **kwargs)
    colorbar(mappable, ax=ax, **kwargs)

    arguments:
        mappable: the image, ContourSet, etc. to which the colorbar applies;
                    this argument is mandatory for the Figure.colorbar
                    method but optional for the pyplot.colorbar function,
                    which sets the default to the current image.

    keyword arguments:
        cax: None | axes object into which the colorbar will be drawn
        ax:  None | parent axes object from which space for a new
                     colorbar axes will be stolen


**kwargs are in two groups:
    axes properties:
%s
    colorbar properties:
%s

If mappable is a ContourSet, its extend kwarg is included automatically.

Note that the shrink kwarg provides a simple way to keep
a vertical colorbar, for example, from being taller than
the axes of the mappable to which the colorbar is attached;
but it is a manual method requiring some trial and error.
If the colorbar is too tall (or a horizontal colorbar is
too wide) use a smaller value of shrink.

For more precise control, you can manually specify the
positions of the axes objects in which the mappable and
the colorbar are drawn.  In this case, do not use any of the
axes properties kwargs.
''' % (make_axes_kw_doc, colormap_kw_doc)



class ColorbarBase(cm.ScalarMappable):
    '''
    Draw a colorbar in an existing axes.

    This is a base class for the Colorbar class, which is
    the basis for the colorbar method and pylab function.

    It is also useful by itself for showing a colormap.  If
    the cmap kwarg is given but boundaries and values are left
    as None, then the colormap will be displayed on a 0-1 scale.
    To show the under- and over-value colors, specify the norm
    as colors.Normalize(clip=False).
    To show the colors versus index instead of on the 0-1 scale,
    use norm=colors.NoNorm.
    '''
    _slice_dict = {'neither': slice(0,1000000),
                   'both': slice(1,-1),
                   'min': slice(1,1000000),
                   'max': slice(0,-1)}

    def __init__(self, ax, cmap=None,
                           norm=None,
                           alpha=1.0,
                           values=None,
                           boundaries=None,
                           orientation='vertical',
                           extend='neither',
                           spacing='uniform',  # uniform or proportional
                           ticks=None,
                           format=None,
                           drawedges=False,
                           filled=True,
                           ):
        self.ax = ax
        if cmap is None: cmap = cm.get_cmap()
        if norm is None: norm = colors.Normalize()
        self.alpha = alpha
        cm.ScalarMappable.__init__(self, cmap=cmap, norm=norm)
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = self._slice_dict[extend]
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self.filled = filled
        self.solids = None
        self.lines = None
        self.set_label('')
        if cbook.iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self.locator = ticks    # Handle default in _ticker()
        if format is None:
            if isinstance(self.norm, colors.LogNorm):
                self.formatter = ticker.LogFormatter()
            else:
                self.formatter = ticker.ScalarFormatter()
        elif cbook.is_string_like(format):
            self.formatter = ticker.FormatStrFormatter(format)
        else:
            self.formatter = format  # Assume it is a Formatter
        # The rest is in a method so we can recalculate when clim changes.
        self.draw_all()

    def draw_all(self):
        '''
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        '''
        self._process_values()
        self._find_range()
        X, Y = self._mesh()
        C = self._values[:,npy.newaxis]
        self._config_axes(X, Y)
        if self.filled:
            self._add_solids(X, Y, C)
        self._set_label()

    def _config_axes(self, X, Y):
        '''
        Make an axes patch and outline.
        '''
        ax = self.ax
        ax.set_frame_on(False)
        ax.set_navigate(False)
        x, y = self._outline(X, Y)
        ax.set_xlim(npy.amin(x), npy.amax(x))
        ax.set_ylim(npy.amin(y), npy.amax(y))
        ax.update_datalim_numerix(x, y)
        self.outline = lines.Line2D(x, y, color=mpl.rcParams['axes.edgecolor'],
                                    linewidth=mpl.rcParams['axes.linewidth'])
        ax.add_artist(self.outline)
        c = mpl.rcParams['axes.facecolor']
        self.patch = patches.Polygon(zip(x,y), edgecolor=c,
                 facecolor=c,
                 linewidth=0.01,
                 zorder=-1)
        ax.add_artist(self.patch)
        ticks, ticklabels, offset_string = self._ticker()
        if self.orientation == 'vertical':
            ax.set_xticks([])
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            ax.yaxis.get_major_formatter().set_offset_string(offset_string)

        else:
            ax.set_yticks([])
            ax.xaxis.set_label_position('bottom')
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.xaxis.get_major_formatter().set_offset_string(offset_string)

    def _set_label(self):
        if self.orientation == 'vertical':
            self.ax.set_ylabel(self._label, **self._labelkw)
        else:
            self.ax.set_xlabel(self._label, **self._labelkw)

    def set_label(self, label, **kw):
        self._label = label
        self._labelkw = kw
        self._set_label()


    def _outline(self, X, Y):
        '''
        Return x, y arrays of colorbar bounding polygon,
        taking orientation into account.
        '''
        N = X.shape[0]
        ii = [0, 1, N-2, N-1, 2*N-1, 2*N-2, N+1, N, 0]
        x = npy.take(npy.ravel(npy.transpose(X)), ii)
        y = npy.take(npy.ravel(npy.transpose(Y)), ii)
        if self.orientation == 'horizontal':
            return y,x
        return x,y

    def _edges(self, X, Y):
        '''
        Return the separator line segments; helper for _add_solids.
        '''
        N = X.shape[0]
        # Using the non-array form of these line segments is much
        # simpler than making them into arrays.
        if self.orientation == 'vertical':
            return [zip(X[i], Y[i]) for i in range(1, N-1)]
        else:
            return [zip(Y[i], X[i]) for i in range(1, N-1)]

    def _add_solids(self, X, Y, C):
        '''
        Draw the colors using pcolor; optionally add separators.
        '''
        ## Change to pcolormesh if/when it is fixed to handle alpha
        ## correctly.
        if self.orientation == 'vertical':
            args = (X, Y, C)
        else:
            args = (npy.transpose(Y), npy.transpose(X), npy.transpose(C))
        kw = {'cmap':self.cmap, 'norm':self.norm,
                    'shading':'flat', 'alpha':self.alpha}
        col = self.ax.pcolor(*args, **kw)
        #self.add_observer(col) # We should observe, not be observed...
        self.solids = col
        if self.drawedges:
            self.dividers = collections.LineCollection(self._edges(X,Y),
                              colors=(mpl.rcParams['axes.edgecolor'],),
                              linewidths=(0.5*mpl.rcParams['axes.linewidth'],)
                              )
            self.ax.add_collection(self.dividers)

    def add_lines(self, levels, colors, linewidths):
        '''
        Draw lines on the colorbar.
        '''
        N = len(levels)
        dummy, y = self._locate(levels)
        if len(y) <> N:
            raise ValueError("levels are outside colorbar range")
        x = npy.array([0.0, 1.0])
        X, Y = npy.meshgrid(x,y)
        if self.orientation == 'vertical':
            xy = [zip(X[i], Y[i]) for i in range(N)]
        else:
            xy = [zip(Y[i], X[i]) for i in range(N)]
        col = collections.LineCollection(xy, linewidths=linewidths)
        self.lines = col
        col.set_color(colors)
        self.ax.add_collection(col)


    def _ticker(self):
        '''
        Return two sequences: ticks (colorbar data locations)
        and ticklabels (strings).
        '''
        locator = self.locator
        formatter = self.formatter
        if locator is None:
            if self.boundaries is None:
                if isinstance(self.norm, colors.NoNorm):
                    nv = len(self._values)
                    base = 1 + int(nv/10)
                    locator = ticker.IndexLocator(base=base, offset=0)
                elif isinstance(self.norm, colors.LogNorm):
                    locator = ticker.LogLocator()
                else:
                    locator = ticker.MaxNLocator()
            else:
                b = self._boundaries[self._inside]
                locator = ticker.FixedLocator(b, nbins=10)
        if isinstance(self.norm, colors.NoNorm):
            intv = transforms.Interval(transforms.Value(self._values[0]),
                                       transforms.Value(self._values[-1]))
        else:
            intv = transforms.Interval(transforms.Value(self.vmin),
                                       transforms.Value(self.vmax))
        locator.set_view_interval(intv)
        locator.set_data_interval(intv)
        formatter.set_view_interval(intv)
        formatter.set_data_interval(intv)
        b = npy.array(locator())
        b, ticks = self._locate(b)
        formatter.set_locs(b)
        ticklabels = [formatter(t) for t in b]
        offset_string = formatter.get_offset()
        return ticks, ticklabels, offset_string

    def _process_values(self, b=None):
        '''
        Set the _boundaries and _values attributes based on
        the input boundaries and values.  Input boundaries can
        be self.boundaries or the argument b.
        '''
        if b is None:
            b = self.boundaries
        if b is not None:
            self._boundaries = npy.array(b)
            if self.values is None:
                self._values = 0.5*(self._boundaries[:-1]
                                        + self._boundaries[1:])
                if isinstance(self.norm, colors.NoNorm):
                    self._values = (self._values + 0.00001).astype(npy.int16)
                return
            self._values = npy.array(self.values)
            return
        if self.values is not None:
            self._values = npy.array(self.values)
            if self.boundaries is None:
                b = npy.zeros(len(self.values)+1, 'd')
                b[1:-1] = 0.5*(self._values[:-1] - self._values[1:])
                b[0] = 2.0*b[1] - b[2]
                b[-1] = 2.0*b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = npy.array(self.boundaries)
            return
        # Neither boundaries nor values are specified;
        # make reasonable ones based on cmap and norm.
        if isinstance(self.norm, colors.NoNorm):
            b = self._uniform_y(self.cmap.N+1) * self.cmap.N - 0.5
            v = npy.zeros((len(b)-1,), dtype=npy.int16)
            v[self._inside] = npy.arange(self.cmap.N, dtype=npy.int16)
            if self.extend in ('both', 'min'):
                v[0] = -1
            if self.extend in ('both', 'max'):
                v[-1] = self.cmap.N
            self._boundaries = b
            self._values = v
            return
        else:
            if not self.norm.scaled():
                self.norm.vmin = 0
                self.norm.vmax = 1
            b = self.norm.inverse(self._uniform_y(self.cmap.N+1))
            if self.extend in ('both', 'min'):
                b[0] = b[0] - 1
            if self.extend in ('both', 'max'):
                b[-1] = b[-1] + 1
        self._process_values(b)

    def _find_range(self):
        '''
        Set vmin and vmax attributes to the first and last
        boundary excluding extended end boundaries.
        '''
        b = self._boundaries[self._inside]
        self.vmin = b[0]
        self.vmax = b[-1]

    def _central_N(self):
        '''number of boundaries *before* extension of ends'''
        nb = len(self._boundaries)
        if self.extend == 'both':
            nb -= 2
        elif self.extend in ('min', 'max'):
            nb -= 1
        return nb

    def _extended_N(self):
        '''
        Based on the colormap and extend variable, return the
        number of boundaries.
        '''
        N = self.cmap.N + 1
        if self.extend == 'both':
            N += 2
        elif self.extend in ('min', 'max'):
            N += 1
        return N

    def _uniform_y(self, N):
        '''
        Return colorbar data coordinates for N uniformly
        spaced boundaries, plus ends if required.
        '''
        if self.extend == 'neither':
            y = npy.linspace(0, 1, N)
        else:
            if self.extend == 'both':
                y = npy.zeros(N + 2, 'd')
                y[0] = -0.05
                y[-1] = 1.05
            elif self.extend == 'min':
                y = npy.zeros(N + 1, 'd')
                y[0] = -0.05
            else:
                y = npy.zeros(N + 1, 'd')
                y[-1] = 1.05
            y[self._inside] = npy.linspace(0, 1, N)
        return y

    def _proportional_y(self):
        '''
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar.
        '''
        y = self.norm(self._boundaries.copy())
        if self.extend in ('both', 'min'):
            y[0] = -0.05
        if self.extend in ('both', 'max'):
            y[-1] = 1.05
        yi = y[self._inside]
        norm = colors.Normalize(yi[0], yi[-1])
        y[self._inside] = norm(yi)
        return y

    def _mesh(self):
        '''
        Return X,Y, the coordinate arrays for the colorbar pcolormesh.
        These are suitable for a vertical colorbar; swapping and
        transposition for a horizontal colorbar are done outside
        this function.
        '''
        x = npy.array([0.0, 1.0])
        if self.spacing == 'uniform':
            y = self._uniform_y(self._central_N())
        else:
            y = self._proportional_y()
        self._y = y
        X, Y = npy.meshgrid(x,y)
        if self.extend in ('min', 'both'):
            X[0,:] = 0.5
        if self.extend in ('max', 'both'):
            X[-1,:] = 0.5
        return X, Y

    def _locate(self, x):
        '''
        Given a possible set of color data values, return the ones
        within range, together with their corresponding colorbar
        data coordinates.
        '''
        if isinstance(self.norm, colors.NoNorm):
            b = self._boundaries
            xn = x
            xout = x
        else:
            # Do calculations using normalized coordinates so
            # as to make the interpolation more accurate.
            b = self.norm(self._boundaries, clip=False).filled()
            # We do our own clipping so that we can allow a tiny
            # bit of slop in the end point ticks to allow for
            # floating point errors.
            xn = self.norm(x, clip=False).filled()
            in_cond = (xn > -0.001) & (xn < 1.001)
            xn = npy.compress(in_cond, xn)
            xout = npy.compress(in_cond, x)
        # The rest is linear interpolation with clipping.
        y = self._y
        N = len(b)
        ii = npy.minimum(npy.searchsorted(b, xn), N-1)
        i0 = npy.maximum(ii - 1, 0)
        #db = b[ii] - b[i0] 
        db = npy.take(b, ii) - npy.take(b, i0)
        db = npy.where(i0==ii, 1.0, db)
        #dy = y[ii] - y[i0]
        dy = npy.take(y, ii) - npy.take(y, i0)
        z = npy.take(y, i0) + (xn-npy.take(b,i0))*dy/db
        return xout, z

    def set_alpha(self, alpha):
        self.alpha = alpha

class Colorbar(ColorbarBase):
    def __init__(self, ax, mappable, **kw):
        mappable.autoscale_None() # Ensure mappable.norm.vmin, vmax
                             # are set when colorbar is called,
                             # even if mappable.draw has not yet
                             # been called.  This will not change
                             # vmin, vmax if they are already set.
        self.mappable = mappable
        kw['cmap'] = mappable.cmap
        kw['norm'] = mappable.norm
        kw['alpha'] = mappable.get_alpha()
        if isinstance(mappable, contour.ContourSet):
            CS = mappable
            kw['boundaries'] = CS._levels
            kw['values'] = CS.cvalues
            kw['extend'] = CS.extend
            #kw['ticks'] = CS._levels
            kw.setdefault('ticks', ticker.FixedLocator(CS.levels, nbins=10))
            kw['filled'] = CS.filled
            ColorbarBase.__init__(self, ax, **kw)
            if not CS.filled:
                self.add_lines(CS)
        else:
            ColorbarBase.__init__(self, ax, **kw)


    def add_lines(self, CS):
        '''
        Add the lines from a non-filled ContourSet to the colorbar.
        '''
        if not isinstance(CS, contour.ContourSet) or CS.filled:
            raise ValueError('add_lines is only for a ContourSet of lines')
        tcolors = [c[0] for c in CS.tcolors]
        tlinewidths = [t[0] for t in CS.tlinewidths]
        # The following was an attempt to get the colorbar lines
        # to follow subsequent changes in the contour lines,
        # but more work is needed: specifically, a careful
        # look at event sequences, and at how
        # to make one object track another automatically.
        #tcolors = [col.get_colors()[0] for col in CS.collections]
        #tlinewidths = [col.get_linewidth()[0] for lw in CS.collections]
        #print 'tlinewidths:', tlinewidths
        ColorbarBase.add_lines(self, CS.levels, tcolors, tlinewidths)

    def notify(self, mappable):
        '''Manually change any contour line colors.  This is called
        when the image or contour plot to which this colorbar belongs
        is changed.
        '''
        cm.ScalarMappable.notify(self, mappable)
        # We are using an ugly brute-force method: clearing and
        # redrawing the whole thing.  The problem is that if any
        # properties have been changed by methods other than the
        # colorbar methods, those changes will be lost.
        self.ax.cla()
        self.draw_all()
        #if self.vmin != self.norm.vmin or self.vmax != self.norm.vmax:
        #    self.ax.cla()
        #    self.draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
            #if self.lines is not None:
            #    tcolors = [c[0] for c in CS.tcolors]
            #    self.lines.set_color(tcolors)
        #Fixme? Recalculate boundaries, ticks if vmin, vmax have changed.
        #Fixme: Some refactoring may be needed; we should not
        # be recalculating everything if there was a simple alpha
        # change.

def make_axes(parent, **kw):
    orientation = kw.setdefault('orientation', 'vertical')
    fraction = kw.pop('fraction', 0.15)
    shrink = kw.pop('shrink', 1.0)
    aspect = kw.pop('aspect', 20)
    #pb = transforms.PBox(parent.get_position())
    pb = transforms.PBox(parent.get_position(original=True))
    if orientation == 'vertical':
        pad = kw.pop('pad', 0.05)
        x1 = 1.0-fraction
        pb1, pbx, pbcb = pb.splitx(x1-pad, x1)
        pbcb.shrink(1.0, shrink).anchor('C')
        anchor = (0.0, 0.5)
        panchor = (1.0, 0.5)
    else:
        pad = kw.pop('pad', 0.15)
        pbcb, pbx, pb1 = pb.splity(fraction, fraction+pad)
        pbcb.shrink(shrink, 1.0).anchor('C')
        aspect = 1.0/aspect
        anchor = (0.5, 1.0)
        panchor = (0.5, 0.0)
    parent.set_position(pb1)
    parent.set_anchor(panchor)
    fig = parent.get_figure()
    cax = fig.add_axes(pbcb)
    cax.set_aspect(aspect, anchor=anchor, adjustable='box')
    return cax, kw
make_axes.__doc__ ='''
    Resize and reposition a parent axes, and return a child
    axes suitable for a colorbar.

    cax, kw = make_axes(parent, **kw)

    Keyword arguments may include the following (with defaults):
        orientation = 'vertical'  or 'horizontal'
    %s

    All but the first of these are stripped from the input kw set.

    Returns (cax, kw), the child axes and the reduced kw dictionary.
    '''  % make_axes_kw_doc


'''
The following does not work correctly.  The problem seems to be that
the transforms work right only when fig.add_axes(rect) is used to
generate the axes, not when the axes object is generated first and
then fig.add_axes(ax) is called.  I don't understand this. - EF

class ColorbarAxes(axes.Axes):
    def __init__(self, parent, **kw):
        orientation = kw.setdefault('orientation', 'vertical')
        fraction = kw.pop('fraction', 0.15)
        shrink = kw.pop('shrink', 1.0)
        aspect = kw.pop('aspect', 20)
        self.cbkw = kw
        pb = transforms.PBox(parent.get_position())
        if orientation == 'vertical':
            pb1, pbcb = pb.splitx(1.0-fraction)
            pbcb.shrink(1.0, shrink).anchor('C')
            anchor = (0.3, 0.5)
            panchor = (0.8, 0.5)
        else:
            pbcb, pb1 = pb.splity(fraction)
            pbcb.shrink(shrink, 1.0).anchor('C')
            aspect = 1.0/aspect
            anchor = (0.5, 0.2)
            panchor = (0.5, 0.8)
        parent.set_position(pb1)
        parent.set_anchor(panchor)
        fig = parent.get_figure()
        axes.Axes.__init__(self, fig, pbcb)
        fig.add_axes(self)
        self.set_aspect(aspect, anchor=anchor, adjustable='box')

'''

