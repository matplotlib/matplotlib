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

import matplotlib.numerix as nx
from matplotlib.mlab import meshgrid, linspace
from matplotlib.numerix.mlab import amin, amax
from matplotlib import colors, cm, ticker
from matplotlib.cbook import iterable, is_string_like
from matplotlib.transforms import Interval, Value, PBox
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.contour import ContourSet
from matplotlib.axes import Axes

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
'''

colorbar_doc = '''
Add a colorbar to a plot.

Function signatures:

    contour(**kwargs)

    contour(mappable, **kwargs)

    contour(mappable, cax, **kwargs)

The optional arguments mappable and cax may be included in the kwargs;
they are image, ContourSet, etc. to which the colorbar applies, and
the axes object in which the colorbar will be drawn.  Defaults are
the current image and a new axes object created next to that image
after resizing the image.

kwargs are in two groups:
    axes properties:
%s
    colorbar properties:
%s

If mappable is a ContourSet, its extend kwarg is included automatically.

''' % (make_axes_kw_doc, colormap_kw_doc)



class ColorbarBase(cm.ScalarMappable):
    '''
    '''
    _slice_dict = {'neither': slice(0,1000000),
                   'both': slice(1,-1),
                   'min': slice(1,1000000),
                   'max': slice(0,-1)}

    def __init__(self, ax, cmap=None,
                           norm=None,
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
        if norm is None: norm = colors.normalize()
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
        if iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=10)
        else:
            self.locator = ticks    # Handle default in _ticker()
        if format is None:
            self.formatter = ticker.ScalarFormatter()
        elif is_string_like(format):
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
        C = self._values[:,nx.NewAxis]
        self._config_axes(X, Y)
        if self.filled:
            self._add_solids(X, Y, C)

    def _config_axes(self, X, Y):
        '''
        Make an axes patch and outline.
        '''
        ax = self.ax
        ax.set_frame_on(False)
        ax.set_navigate(False)
        x, y = self._outline(X, Y)
        ax.set_xlim(amin(x), amax(x))
        ax.set_ylim(amin(y), amax(y))
        ax.update_datalim_numerix(x, y)
        self.outline = Line2D(x, y, color=rcParams['axes.edgecolor'],
                                    linewidth=rcParams['axes.linewidth'])
        ax.add_artist(self.outline)
        c = rcParams['axes.facecolor']
        self.patch = Polygon(zip(x,y), edgecolor=c,
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


    def _outline(self, X, Y):
        '''
        Return x, y arrays of colorbar bounding polygon,
        taking orientation into account.
        '''
        N = nx.shape(X)[0]
        ii = [0, 1, N-2, N-1, 2*N-1, 2*N-2, N+1, N, 0]
        x = nx.take(nx.ravel(nx.transpose(X)), ii)
        y = nx.take(nx.ravel(nx.transpose(Y)), ii)
        if self.orientation == 'horizontal':
            return y,x
        return x,y

    def _edges(self, X, Y):
        '''
        Return the separator line segments; helper for _add_solids.
        '''
        N = nx.shape(X)[0]
        if self.orientation == 'vertical':
            return [zip(X[i], Y[i]) for i in range(1, N-1)]
        else:
            return [zip(Y[i], X[i]) for i in range(1, N-1)]

    def _add_solids(self, X, Y, C):
        '''
        Draw the colors using pcolormesh; optionally add separators.
        '''
        if self.orientation == 'vertical':
            args = (X, Y, C)
        else:
            args = (nx.transpose(Y), nx.transpose(X), nx.transpose(C))
        kw = {'cmap':self.cmap, 'norm':self.norm, 'shading':'flat'}
        col = self.ax.pcolormesh(*args, **kw)
        self.add_observer(col)
        self.solids = col
        if self.drawedges:
            self.dividers = LineCollection(self._edges(X,Y),
                                           colors=(rcParams['axes.edgecolor'],),
                                           linewidths=(0.5*rcParams['axes.linewidth'],)
                                           )
            self.ax.add_collection(self.dividers)

    def add_lines(self, levels, colors, linewidths):
        '''
        Draw lines on the colorbar.
        '''
        N = len(levels)
        y = self._locate(levels)
        x = nx.array([0.0, 1.0])
        X, Y = meshgrid(x,y)
        if self.orientation == 'vertical':
            xy = [zip(X[i], Y[i]) for i in range(N)]
        else:
            xy = [zip(Y[i], X[i]) for i in range(N)]
        col = LineCollection(xy, linewidths=linewidths)
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
                locator = ticker.MaxNLocator()
            else:
                b = self._boundaries[self._inside]
                locator = ticker.FixedLocator(b, nbins=10)
        intv = Interval(Value(self.vmin), Value(self.vmax))
        locator.set_view_interval(intv)
        locator.set_data_interval(intv)
        formatter.set_view_interval(intv)
        formatter.set_data_interval(intv)
        b = nx.array(locator())
        eps = 0.001 * (self.vmax - self.vmin)
        b = nx.compress((b >= self.vmin-eps) & (b <= self.vmax+eps), b)
        ticks = self._locate(b)
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
            self._boundaries = nx.array(b)
            if self.values is None:
                self._values = 0.5*(self._boundaries[:-1]
                                        + self._boundaries[1:])
                return
            self._values = nx.array(self.values)
            return
        if self.values is not None:
            self._values = nx.array(self.values)
            if self.boundaries is None:
                b = nx.zeros(len(self.values)+1, 'd')
                b[1:-1] = 0.5*(self._values[:-1] - self._values[1:])
                b[0] = 2.0*b[1] - b[2]
                b[-1] = 2.0*b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = nx.array(self.boundaries)
            return
        dv = self.norm.vmax - self.norm.vmin
        b = self.norm.vmin + dv * self._uniform_y(self.cmap.N+1)
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
            y = linspace(0, 1, N)
        else:
            if self.extend == 'both':
                y = nx.zeros(N + 2, 'd')
                y[0] = -0.05
                y[-1] = 1.05
            elif self.extend == 'min':
                y = nx.zeros(N + 1, 'd')
                y[0] = -0.05
            else:
                y = nx.zeros(N + 1, 'd')
                y[-1] = 1.05
            y[self._inside] = linspace(0, 1, N)
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
        norm = colors.normalize(yi[0], yi[-1])
        y[self._inside] = norm(yi)
        return y

    def _mesh(self):
        '''
        Return X,Y, the coordinate arrays for the colorbar pcolormesh.
        These are suitable for a vertical colorbar; swapping and
        transposition for a horizontal colorbar are done outside
        this function.
        '''
        x = nx.array([0.0, 1.0])
        if self.spacing == 'uniform':
            y = self._uniform_y(self._central_N())
        else:
            y = self._proportional_y()
        self._y = y
        X, Y = meshgrid(x,y)
        if self.extend in ('min', 'both'):
            X[0,:] = 0.5
        if self.extend in ('max', 'both'):
            X[-1,:] = 0.5
        return X, Y

    def _locate(self, x):
        '''
        Return the colorbar data coordinate(s) corresponding to the color
        value(s) in scalar or array x.
        Used for tick positioning.
        '''
        b = self._boundaries
        y = self._y
        N = len(b)
        ii = nx.minimum(nx.searchsorted(b, x), N-1)
        isscalar = False
        if not iterable(ii):
            isscalar = True
            ii = nx.array((ii,))
        i0 = nx.maximum(ii - 1, 0)
        #db = b[ii] - b[i0]
        db = nx.take(b, ii) - nx.take(b, i0)
        db = nx.where(i0==ii, 1.0, db)
        #dy = y[ii] - y[i0]
        dy = nx.take(y, ii) - nx.take(y, i0)
        z = nx.take(y, i0) + (x-nx.take(b,i0))*dy/db
        if isscalar:
            z = z[0]
        return z

class Colorbar(ColorbarBase):
    def __init__(self, ax, mappable, **kw):
        self.mappable = mappable
        if isinstance(mappable, ContourSet):
            CS = mappable
            kw['cmap'] = CS.cmap
            kw['norm'] = CS.norm
            kw['boundaries'] = CS._levels
            kw['values'] = CS.cvalues
            kw['extend'] = CS.extend
            #kw['ticks'] = CS._levels
            kw.setdefault('ticks', CS._levels)
            kw['filled'] = CS.filled
            ColorbarBase.__init__(self, ax, **kw)
            if not CS.filled:
                self.add_lines(CS)
        else:
            kw['cmap'] = mappable.cmap
            kw['norm'] = mappable.norm
            ColorbarBase.__init__(self, ax, **kw)


    def add_lines(self, CS):
        '''
        Add the lines from a non-filled ContourSet to the colorbar.
        '''
        if not isinstance(CS, ContourSet) or CS.filled:
            raise ValueError('add_lines is only for a ContourSet of lines')
        tcolors = [c[0] for c in CS.tcolors]
        tlinewidths = [t[0] for t in CS.tlinewidths]
        ColorbarBase.add_lines(self, CS.levels, tcolors, tlinewidths)

    def notify(self, mappable):
        '''Manually change any contour line colors.  This is called
        when the image or contour plot to which this colorbar belongs
        is changed.
        '''
        cm.ScalarMappable.notify(self, mappable)
        if self.vmin != self.norm.vmin or self.vmax != self.norm.vmax:
            self.ax.cla()
            self.draw_all()
        if isinstance(self.mappable, ContourSet):
            CS = self.mappable
            if self.lines is not None:
                tcolors = [c[0] for c in CS.tcolors]
                self.lines.set_color(tcolors)
        #Fixme? Recalculate boundaries, ticks if vmin, vmax have changed.


def make_axes(parent, **kw):
    orientation = kw.setdefault('orientation', 'vertical')
    fraction = kw.pop('fraction', 0.15)
    shrink = kw.pop('shrink', 1.0)
    aspect = kw.pop('aspect', 20)
    #pb = PBox(parent.get_position())
    pb = PBox(parent.get_position(original=True))
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

class ColorbarAxes(Axes):
    def __init__(self, parent, **kw):
        orientation = kw.setdefault('orientation', 'vertical')
        fraction = kw.pop('fraction', 0.15)
        shrink = kw.pop('shrink', 1.0)
        aspect = kw.pop('aspect', 20)
        self.cbkw = kw
        pb = PBox(parent.get_position())
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
        Axes.__init__(self, fig, pbcb)
        fig.add_axes(self)
        self.set_aspect(aspect, anchor=anchor, adjustable='box')

'''

