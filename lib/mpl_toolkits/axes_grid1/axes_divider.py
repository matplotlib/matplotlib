"""
The axes_divider module provide helper classes to adjust the positions of
multiple axes at the drawing time.

 Divider: this is the class that is used calculates the axes
    position. It divides the given renctangular area into several sub
    rectangles. You intialize the divider by setting the horizontal
    and vertical list of sizes that the division will be based on. You
    then use the new_locator method, whose return value is a callable
    object that can be used to set the axes_locator of the axes.

"""

import matplotlib.transforms as mtransforms

from matplotlib.axes import SubplotBase

import new

import axes_size as Size


class Divider(object):
    """
    This is the class that is used calculates the axes position. It
    divides the given renctangular area into several
    sub-rectangles. You intialize the divider by setting the
    horizontal and vertical lists of sizes
    (:mod:`mpl_toolkits.axes_grid.axes_size`) that the division will
    be based on. You then use the new_locator method to create a
    callable object that can be used to as the axes_locator of the
    axes.
    """


    def __init__(self, fig, pos, horizontal, vertical, aspect=None, anchor="C"):
        """
        :param fig: matplotlib figure
        :param pos: position (tuple of 4 floats) of the rectangle that
                    will be divided.
        :param horizontal: list of sizes
                    (:mod:`~mpl_toolkits.axes_grid.axes_size`)
                    for horizontal division
        :param vertical: list of sizes
                    (:mod:`~mpl_toolkits.axes_grid.axes_size`)
                    for vertical division
        :param aspect: if True, the overall rectalngular area is reduced
                    so that the relative part of the horizontal and
                    vertical scales have same scale.
        :param anchor: Detrmine how the reduced rectangle is placed
                       when aspect is True,
        """

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
            _rs, _as = s.get_size(renderer)
            rs_sum += _rs
            as_sum += _as

        if rs_sum != 0.:
            k = (total_size - as_sum) / rs_sum
            return k
        else:
            return 0.


    @staticmethod
    def _calc_offsets(l, k, renderer):

        offsets = [0.]

        for s in l:
            _rs, _as = s.get_size(renderer)
            offsets.append(offsets[-1] + _rs*k + _as)

        return offsets


    def set_position(self, pos):
        """
        set the position of the rectangle.

        :param pos: position (tuple of 4 floats) of the rectangle that
                    will be divided.
        """
        self._pos = pos

    def get_position(self):
        "return the position of the rectangle."
        return self._pos

    def set_anchor(self, anchor):
        """
        :param anchor: anchor position

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

    def get_anchor(self):
        "return the anchor"
        return self._anchor

    def set_horizontal(self, h):
        """
        :param horizontal: list of sizes
                    (:mod:`~mpl_toolkits.axes_grid.axes_size`)
                    for horizontal division
        """
        self._horizontal = h


    def get_horizontal(self):
        "return horizontal sizes"
        return self._horizontal

    def set_vertical(self, v):
        """
        :param horizontal: list of sizes
                    (:mod:`~mpl_toolkits.axes_grid.axes_size`)
                    for horizontal division
        """
        self._vertical = v

    def get_vertical(self):
        "return vertical sizes"
        return self._vertical


    def set_aspect(self, aspect=False):
        """
        :param anchor: True or False
        """
        self._aspect = aspect

    def get_aspect(self):
        "return aspect"
        return self._aspect


    def locate(self, nx, ny, nx1=None, ny1=None, renderer=None):
        """

        :param nx, nx1: Integers specifying the column-position of the
          cell. When nx1 is None, a single nx-th column is
          specified. Otherwise location of columns spanning between nx
          to nx1 (but excluding nx1-th column) is specified.

        :param ny, ny1: same as nx and nx1, but for row positions.
        """


        figW,figH = self._fig.get_size_inches()
        x, y, w, h = self.get_position()

        k_h = self._calc_k(self._horizontal, figW*w, renderer)
        k_v = self._calc_k(self._vertical, figH*h, renderer)

        if self.get_aspect():
            k = min(k_h, k_v)
            ox = self._calc_offsets(self._horizontal, k, renderer)
            oy = self._calc_offsets(self._vertical, k, renderer)

            ww = (ox[-1] - ox[0])/figW
            hh = (oy[-1] - oy[0])/figH
            pb = mtransforms.Bbox.from_bounds(x, y, w, h)
            pb1 = mtransforms.Bbox.from_bounds(x, y, ww, hh)
            pb1_anchored = pb1.anchored(self.get_anchor(), pb)
            x0, y0 = pb1_anchored.x0, pb1_anchored.y0

        else:
            ox = self._calc_offsets(self._horizontal, k_h, renderer)
            oy = self._calc_offsets(self._vertical, k_v, renderer)
            x0, y0 = x, y


        if nx1 is None:
            nx1=nx+1
        if ny1 is None:
            ny1=ny+1

        x1, w1 = x0 + ox[nx]/figW, (ox[nx1] - ox[nx])/figW
        y1, h1 = y0 + oy[ny]/figH, (oy[ny1] - oy[ny])/figH

        return mtransforms.Bbox.from_bounds(x1, y1, w1, h1)


    def new_locator(self, nx, ny, nx1=None, ny1=None):
        """
        returns a new locator
        (:class:`mpl_toolkits.axes_grid.axes_divider.AxesLocator`) for
        specified cell.

        :param nx, nx1: Integers specifying the column-position of the
          cell. When nx1 is None, a single nx-th column is
          specified. Otherwise location of columns spanning between nx
          to nx1 (but excluding nx1-th column) is specified.

        :param ny, ny1: same as nx and nx1, but for row positions.
        """
        return AxesLocator(self, nx, ny, nx1, ny1)



class AxesLocator(object):
    """
    A simple callable object, initiallized with AxesDivider class,
    returns the position and size of the given cell.
    """
    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):
        """
        :param axes_divider: An instance of AxesDivider class.

        :param nx, nx1: Integers specifying the column-position of the
          cell. When nx1 is None, a single nx-th column is
          specified. Otherwise location of columns spanning between nx
          to nx1 (but excluding nx1-th column) is is specified.

        :param ny, ny1: same as nx and nx1, but for row positions.
        """
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

        return self._axes_divider.locate(self._nx + _xrefindex,
                                         self._ny + _yrefindex,
                                         self._nx1 + _xrefindex,
                                         self._ny1 + _yrefindex,
                                         renderer)



class SubplotDivider(Divider):
    """
    The Divider class whose rectangle area is specified as a subplot grometry.
    """


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
        "return the bounds of the subplot box"
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
    """
    Divider based on the pre-existing axes.
    """

    def __init__(self, axes):
        """
        :param axes: axes
        """
        self._axes = axes
        self._xref = Size.AxesX(axes)
        self._yref = Size.AxesY(axes)
        Divider.__init__(self, fig=axes.get_figure(), pos=None,
                         horizontal=[self._xref], vertical=[self._yref],
                         aspect=None, anchor="C")

    def _get_new_axes(self, **kwargs):
        axes = self._axes

        axes_class = kwargs.pop("axes_class", None)

        if axes_class is None:
            if isinstance(axes, SubplotBase):
                axes_class = axes._axes_class
            else:
                axes_class = type(axes)

        ax = axes_class(axes.get_figure(),
                        axes.get_position(original=True), **kwargs)

        return ax


    def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
        """
        Add a new axes on the right (or left) side of the main axes.

        :param size: A width of the axes. A :mod:`~mpl_toolkits.axes_grid.axes_size`
          instance or if float or string is given, *from_any*
          fucntion is used to create one, with *ref_size* set to AxesX instance
          of the current axes.
        :param pad: pad between the axes. It takes same argument as *size*.
        :param pack_start: If False, the new axes is appended at the end
          of the list, i.e., it became the right-most axes. If True, it is
          inseted at the start of the list, and becomes the left-most axes.

        All extra keywords argument is passed to when creating a axes.
        if *axes_class* is given, the new axes will be created as an
        instance of the given class. Otherwise, the same class of the
        main axes will be used.  if Not provided

        """

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
            self._horizontal.insert(0, size)
            self._xrefindex += 1
            locator = self.new_locator(nx=0, ny=0)
        else:
            self._horizontal.append(size)
            locator = self.new_locator(nx=len(self._horizontal)-1, ny=0)

        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)

        return ax

    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
        """
        Add a new axes on the top (or bottom) side of the main axes.

        :param size: A height of the axes. A :mod:`~mpl_toolkits.axes_grid.axes_size`
          instance or if float or string is given, *from_any*
          fucntion is used to create one, with *ref_size* set to AxesX instance
          of the current axes.
        :param pad: pad between the axes. It takes same argument as *size*.
        :param pack_start: If False, the new axes is appended at the end
          of the list, i.e., it became the top-most axes. If True, it is
          inseted at the start of the list, and becomes the bottom-most axes.

        All extra keywords argument is passed to when creating a axes.
        if *axes_class* is given, the new axes will be created as an
        instance of the given class. Otherwise, the same class of the
        main axes will be used.  if Not provided

        """

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
            self._vertical.insert(0, size)
            self._yrefindex += 1
            locator = self.new_locator(nx=0, ny=0)
        else:
            self._vertical.append(size)
            locator = self.new_locator(nx=0, ny=len(self._vertical)-1)

        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)

        return ax


    def append_axes(self, position, size, pad=None, add_to_figure=True,
                    **kwargs):
        """
        create an axes at the given *position* with the same height
        (or width) of the main axes.

         *position*
           ["left"|"right"|"bottom"|"top"]

         *size* and *pad* should be axes_grid.axes_size compatible.
        """

        if position == "left":
            ax = self.new_horizontal(size, pad, pack_start=True, **kwargs)
        elif position == "right":
            ax = self.new_horizontal(size, pad, pack_start=False, **kwargs)
        elif position == "bottom":
            ax = self.new_vertical(size, pad, pack_start=True, **kwargs)
        elif position == "top":
            ax = self.new_vertical(size, pad, pack_start=False, **kwargs)
        else:
            raise ValueError("the position must be one of left, right, bottom, or top")

        if add_to_figure:
            self._fig.add_axes(ax)
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
            self._axes_class.apply_aspect(self, position)
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

#if hasattr(maxes.Axes, "get_axes_locator"):
#    LocatableAxes = maxes.Axes
#else:

def make_axes_locatable(axes):
    if not hasattr(axes, "set_axes_locator"):
        new_class = locatable_axes_factory(type(axes))
        axes.__class__ = new_class

    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    return divider


#from matplotlib.axes import Axes
from mpl_axes import Axes
LocatableAxes = locatable_axes_factory(Axes)



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
    plt.show()


def demo_fixed_size_axes():
    import matplotlib.pyplot as plt

    fig2 = plt.figure(2, (6, 6))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(1.0), Size.Fixed(4.5)]
    v = [Size.Fixed(0.7), Size.Fixed(5.)]

    divider = Divider(fig2, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    # the width and height of the rectangle is ignored.

    ax = LocatableAxes(fig2, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig2.add_axes(ax)

    ax.plot([1,2,3])

    plt.draw()
    plt.show()
    #plt.colorbar(im, cax=ax_cb)





if __name__ == "__main__":
    demo_locatable_axes()
    demo_fixed_size_axes()
