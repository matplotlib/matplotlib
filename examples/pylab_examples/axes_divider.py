
import matplotlib.axes as maxes
import matplotlib.transforms as mtransforms

import matplotlib.cbook as cbook

import new


class Size(object):

    @classmethod
    def from_any(self, size, fraction_ref=None):
        if cbook.is_numlike(size):
            return Size.Fixed(size)
        elif cbook.is_string_like(size):
            if size[-1] == "%":
                return Size.Fraction(fraction_ref, float(size[:-1])/100.)

        raise ValueError("")



    class _Base(object):
        pass

    class Fixed(_Base):
        def __init__(self, fixed_size):
            self._fixed_size = fixed_size

        def get_size(self, renderer):
            rel_size = 0.
            abs_size = self._fixed_size
            return rel_size, abs_size

    class Scalable(_Base):
        def __init__(self, scalable_size):
            self._scalable_size = scalable_size

        def get_size(self, renderer):
            rel_size = self._scalable_size
            abs_size = 0.
            return rel_size, abs_size


    class AxesX(_Base):
        def __init__(self, axes, aspect=1.):
            self._axes = axes
            self._aspect = aspect

        def get_size(self, renderer):
            l1, l2 = self._axes.get_xlim()
            rel_size = abs(l2-l1)*self._aspect
            abs_size = 0.
            return rel_size, abs_size

    class AxesY(_Base):
        def __init__(self, axes, aspect=1.):
            self._axes = axes
            self._aspect = aspect

        def get_size(self, renderer):
            l1, l2 = self._axes.get_ylim()
            rel_size = abs(l2-l1)*self._aspect
            abs_size = 0.
            return rel_size, abs_size


    class MaxExtent(_Base):
        def __init__(self, artist_list, w_or_h):
            self._artist_list = artist_list

            if w_or_h not in ["width", "height"]:
                raise ValueError()

            self._w_or_h = w_or_h

        def add_artist(self, a):
            self._artist_list.append(a)

        def get_size(self, renderer):
            rel_size = 0.
            w_list, h_list = [], []
            for a in self._artist_list:
                bb = a.get_window_extent(renderer)
                w_list.append(bb.width)
                h_list.append(bb.height)
            dpi = a.get_figure().get_dpi()
            if self._w_or_h == "width":
                abs_size = max(w_list)/dpi
            elif self._w_or_h == "height":
                abs_size = max(h_list)/dpi

            return rel_size, abs_size

    class Fraction(_Base):
        def __init__(self, size, fraction):
            self._size = size
            self._fraction = fraction

        def get_size(self, renderer):
            r, a = self._size.get_size(renderer)
            rel_size = r*self._fraction
            abs_size = a*self._fraction
            return rel_size, abs_size

    class Padded(_Base):
        def __init__(self, size, pad):
            self._size = size
            self._pad = pad

        def get_size(self, renderer):
            r, a = self._size.get_size(renderer)
            rel_size = r
            abs_size = a + self._pad
            return rel_size, abs_size



class AxesLocator(object):
    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):

        self._axes_divider = axes_divider

        _xrefindex = axes_divider._xrefindex
        _yrefindex = axes_divider._yrefindex

        self._nx, self._ny = nx - _xrefindex, ny - _yrefindex

        if nx1 is None:
            nx1 = nx+1
        if ny1 is None:
            ny1 = ny+1

        self._nx1 = nx1 - _xrefindex
        self._ny1 = ny1 - _yrefindex


    def __call__(self, axes, renderer):

        _xrefindex = self._axes_divider._xrefindex
        _yrefindex = self._axes_divider._yrefindex

        return self._axes_divider.locate(self._nx + _xrefindex, self._ny + _yrefindex,
                                         self._nx1 + _xrefindex, self._ny1 + _yrefindex,
                                         renderer)


class Divider(object):

    def __init__(self, fig, pos, horizontal, vertical, aspect=None, anchor="C"):
        self._fig = fig
        self._pos = pos
        self._horizontal = horizontal
        self._vertical = vertical
        self._anchor = anchor
        self._aspect = aspect
        self._xrefindex = 0
        self._yrefindex = 0


    @staticmethod
    def _calc_k(l, total_size, renderer):

        rs_sum, as_sum = 0., 0.

        for s in l:
            rs, as = s.get_size(renderer)
            rs_sum += rs
            as_sum += as

        k = (total_size - as_sum) / rs_sum
        return k


    @staticmethod
    def _calc_offsets(l, k, renderer):

        offsets = [0.]

        for s in l:
            rs, as = s.get_size(renderer)
            offsets.append(offsets[-1] + rs*k + as)

        return offsets


    def set_position(self, pos):
        self._pos = pos

    def get_position(self):
        return self._pos

    def set_anchor(self, anchor):
        """
        *anchor*

          =====  ============
          value  description
          =====  ============
          'C'    Center
          'SW'   bottom left
          'S'    bottom
          'SE'   bottom right
          'E'    right
          'NE'   top right
          'N'    top
          'NW'   top left
          'W'    left
          =====  ============

        """
        if anchor in mtransforms.Bbox.coefs.keys() or len(anchor) == 2:
            self._anchor = anchor
        else:
            raise ValueError('argument must be among %s' %
                                ', '.join(mtransforms.BBox.coefs.keys()))


    def set_horizontal(self, h):
        self._horizontal = h

    def get_horizontal(self):
        return self._horizontal

    def set_vertical(self, v):
        self._vertical = v

    def get_vertical(self):
        return self._vertical


    def get_anchor(self):
        return self._anchor


    def set_aspect(self, aspect=False):
        """
        *aspect* : True or False
        """
        self._aspect = aspect

    def get_aspect(self):
        return self._aspect


    def locate(self, nx, ny, nx1=None, ny1=None, renderer=None):


        figW,figH = self._fig.get_size_inches()
        x, y, w, h = self.get_position()

        k_h = self._calc_k(self._horizontal, figW*w, renderer)
        k_v = self._calc_k(self._vertical, figH*h, renderer)

        if self.get_aspect():
            k = min(k_h, k_v)
            ox = self._calc_offsets(self._horizontal, k, renderer)
            oy = self._calc_offsets(self._vertical, k, renderer)
        else:
            ox = self._calc_offsets(self._horizontal, k_h, renderer)
            oy = self._calc_offsets(self._vertical, k_v, renderer)


        ww = (ox[-1] - ox[0])/figW
        hh = (oy[-1] - oy[0])/figH
        pb = mtransforms.Bbox.from_bounds(x, y, w, h)
        pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
        pb1_anchored = pb1.anchored(self.get_anchor(), pb)

        if nx1 is None:
            nx1=nx+1
        if ny1 is None:
            ny1=ny+1

        x0, y0 = pb1_anchored.x0, pb1_anchored.y0
        x1, w1 = x0 + ox[nx]/figW, (ox[nx1] - ox[nx])/figW
        y1, h1 = y0 + oy[ny]/figH, (oy[ny1] - oy[ny])/figH

        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


    def new_locator(self, nx, ny, nx1=None, ny1=None):
        return AxesLocator(self, nx, ny, nx1, ny1)


class SubplotDivider(Divider):

    def __init__(self, fig, *args, **kwargs):
        """
        *fig* is a :class:`matplotlib.figure.Figure` instance.

        *args* is the tuple (*numRows*, *numCols*, *plotNum*), where
        the array of subplots in the figure has dimensions *numRows*,
        *numCols*, and where *plotNum* is the number of the subplot
        being created.  *plotNum* starts at 1 in the upper left
        corner and increases to the right.

        If *numRows* <= *numCols* <= *plotNum* < 10, *args* can be the
        decimal integer *numRows* * 100 + *numCols* * 10 + *plotNum*.
        """

        self.figure = fig

        if len(args)==1:
            s = str(args[0])
            if len(s) != 3:
                raise ValueError('Argument to subplot must be a 3 digits long')
            rows, cols, num = map(int, s)
        elif len(args)==3:
            rows, cols, num = args
        else:
            raise ValueError(  'Illegal argument to subplot')


        total = rows*cols
        num -= 1    # convert from matlab to python indexing
                    # ie num in range(0,total)
        if num >= total:
            raise ValueError( 'Subplot number exceeds total subplots')
        self._rows = rows
        self._cols = cols
        self._num = num

        self.update_params()

        pos = self.figbox.bounds
        horizontal = kwargs.pop("horizontal", [])
        vertical = kwargs.pop("vertical", [])
        aspect = kwargs.pop("aspect", None)
        anchor = kwargs.pop("anchor", "C")

        if kwargs:
            raise Exception("")

        Divider.__init__(self, fig, pos, horizontal, vertical,
                         aspect=aspect, anchor=anchor)


    def get_position(self):
        self.update_params()
        return self.figbox.bounds


    def update_params(self):
        'update the subplot position from fig.subplotpars'

        rows = self._rows
        cols = self._cols
        num = self._num

        pars = self.figure.subplotpars
        left = pars.left
        right = pars.right
        bottom = pars.bottom
        top = pars.top
        wspace = pars.wspace
        hspace = pars.hspace
        totWidth = right-left
        totHeight = top-bottom

        figH = totHeight/(rows + hspace*(rows-1))
        sepH = hspace*figH

        figW = totWidth/(cols + wspace*(cols-1))
        sepW = wspace*figW

        rowNum, colNum =  divmod(num, cols)

        figBottom = top - (rowNum+1)*figH - rowNum*sepH
        figLeft = left + colNum*(figW + sepW)

        self.figbox = mtransforms.Bbox.from_bounds(figLeft, figBottom,
                                                   figW, figH)

class AxesDivider(Divider):


    def __init__(self, axes):
        self._axes = axes
        self._xref = Size.AxesX(axes)
        self._yref = Size.AxesY(axes)
        Divider.__init__(self, fig=axes.get_figure(), pos=None,
                         horizontal=[self._xref], vertical=[self._yref],
                         aspect=None, anchor="C")

    def new_horizontal(self, size, pad=None, pack_start=False):

        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad,
                                    fraction_ref=self._xref)
            if pack_start:
                self._horizontal.insert(0, pad)
                self._xrefindex += 1
            else:
                self._horizontal.append(pad)

        if not isinstance(size, Size._Base):
            size = Size.from_any(size,
                                 fraction_ref=self._xref)

        if pack_start:
            self._horizontal.insert(0, pad)
            self._xrefindex += 1
            locator = self.new_locator(nx=0, ny=0)
        else:
            self._horizontal.append(size)
            locator = self.new_locator(nx=len(self._horizontal)-1, ny=0)

        ax = LocatableAxes(self._axes.get_figure(),
                           self._axes.get_position(original=True))
        locator = self.new_locator(nx=len(self._horizontal)-1, ny=0)
        ax.set_axes_locator(locator)

        return ax

    def new_vertical(self, size, pad=None, pack_start=False):

        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad,
                                    fraction_ref=self._yref)
            if pack_start:
                self._vertical.insert(0, pad)
                self._yrefindex += 1
            else:
                self._vertical.append(pad)

        if not isinstance(size, Size._Base):
            size = Size.from_any(size,
                                 fraction_ref=self._yref)

        if pack_start:
            self._vertical.insert(0, pad)
            self._yrefindex += 1
            locator = self.new_locator(nx=0, ny=0)
        else:
            self._vertical.append(size)
            locator = self.new_locator(nx=0, ny=len(self._vertical)-1)

        ax = LocatableAxes(self._axes.get_figure(),
                           self._axes.get_position(original=True))
        ax.set_axes_locator(locator)

        return ax


    def get_aspect(self):
        if self._aspect is None:
            aspect = self._axes.get_aspect()
            if aspect == "auto":
                return False
            else:
                return True
        else:
            return self._aspect

    def get_position(self):
        if self._pos is None:
            bbox = self._axes.get_position(original=True)
            return bbox.bounds
        else:
            return self._pos

    def get_anchor(self):
        if self._anchor is None:
            return self._axes.get_anchor()
        else:
            return self._anchor



class LocatableAxesBase:
    def __init__(self, *kl, **kw):

        self._axes_class.__init__(self, *kl, **kw)

        self._locator = None
        self._locator_renderer = None

    def set_axes_locator(self, locator):
        self._locator = locator

    def get_axes_locator(self):
        return self._locator

    def apply_aspect(self, position=None):

        if self.get_axes_locator() is None:
            self._axes_class.apply_apsect(self, position)
        else:
            pos = self.get_axes_locator()(self, self._locator_renderer)
            self._axes_class.apply_aspect(self, position=pos)


    def draw(self, renderer=None, inframe=False):

        self._locator_renderer = renderer

        self._axes_class.draw(self, renderer, inframe)



_locatableaxes_classes = {}
def locatable_axes_factory(axes_class):

    new_class = _locatableaxes_classes.get(axes_class)
    if new_class is None:
        new_class = new.classobj("Locatable%s" % (axes_class.__name__),
                                 (LocatableAxesBase, axes_class),
                                 {'_axes_class': axes_class})
        _locatableaxes_classes[axes_class] = new_class

    return new_class

if hasattr(maxes.Axes, "get_axes_locator"):
    LocatableAxes = maxes.Axes
else:
    LocatableAxes = locatable_axes_factory(maxes.Axes)


def make_axes_locatable(axes):
    if not hasattr(axes, "set_axes_locator"):
        new_class = locatable_axes_factory(type(axes))
        axes.__class__ = new_class

    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    return divider


def get_demo_image():
    # prepare image
    delta = 0.5

    extent = (-3,4,-4,3)
    import numpy as np
    x = np.arange(-3.0, 4.001, delta)
    y = np.arange(-4.0, 3.001, delta)
    X, Y = np.meshgrid(x, y)
    import matplotlib.mlab as mlab
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = (Z1 - Z2) * 10

    return Z, extent

def demo_locatable_axes():
    import matplotlib.pyplot as plt

    fig1 = plt.figure(1, (6, 6))
    fig1.clf()

    ## PLOT 1
    # simple image & colorbar
    ax = fig1.add_subplot(2, 2, 1)

    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    cb = plt.colorbar(im)
    plt.setp(cb.ax.get_yticklabels(), visible=False)
    

    ## PLOT 2 
    # image and colorbar whose location is adjusted in the drawing time.
    # a hard way

    divider = SubplotDivider(fig1, 2, 2, 2, aspect=True)

    # axes for image
    ax = LocatableAxes(fig1, divider.get_position())

    # axes for coloarbar
    ax_cb = LocatableAxes(fig1, divider.get_position())

    h = [Size.AxesX(ax), # main axes
         Size.Fixed(0.05), # padding, 0.1 inch
         Size.Fixed(0.2), # colorbar, 0.3 inch
         ]

    v = [Size.AxesY(ax)]

    divider.set_horizontal(h)
    divider.set_vertical(v)

    ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    ax_cb.set_axes_locator(divider.new_locator(nx=2, ny=0))

    fig1.add_axes(ax)
    fig1.add_axes(ax_cb)

    ax_cb.yaxis.set_ticks_position("right")

    Z, extent = get_demo_image()

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    plt.colorbar(im, cax=ax_cb)
    plt.setp(ax_cb.get_yticklabels(), visible=False)

    plt.draw()
    #plt.colorbar(im, cax=ax_cb)


    ## PLOT 3
    # image and colorbar whose location is adjusted in the drawing time.
    # a easy way

    ax = fig1.add_subplot(2, 2, 3)
    divider = make_axes_locatable(ax)
    
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig1.add_axes(ax_cb)

    im = ax.imshow(Z, extent=extent, interpolation="nearest")
    plt.colorbar(im, cax=ax_cb)
    plt.setp(ax_cb.get_yticklabels(), visible=False)


    ## PLOT 4
    # two images side by sied with fixed padding.

    ax = fig1.add_subplot(2, 2, 4)
    divider = make_axes_locatable(ax)

    ax2 = divider.new_horizontal(size="100%", pad=0.05)
    fig1.add_axes(ax2)

    ax.imshow(Z, extent=extent, interpolation="nearest")
    ax2.imshow(Z, extent=extent, interpolation="nearest")
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.draw()

if __name__ == "__main__":
    demo_locatable_axes()
