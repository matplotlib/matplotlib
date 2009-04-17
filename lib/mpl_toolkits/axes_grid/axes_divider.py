"""
The axes_divider module provide helper classes to adjust the axes
positions of set of images in the drawing time.

 Size: This provides a classese of units that the size of each axes
    will be determined. For example, you can specify a fixed size

 Divider: this is the class that uis used calculates the axes
    position. It divides the given renctangular area into several
    areas. You intialize the divider by setting the horizontal and
    vertical list of sizes that the division will be based on. You
    then use the new_locator method, whose return value is a callable
    object that can be used to set the axes_locator of the axes.

"""

import matplotlib.axes as maxes
import matplotlib.transforms as mtransforms

import matplotlib.cbook as cbook
from matplotlib.axes import SubplotBase

import new


class Size(object):
    """
    provides a classese of units that will be used with AxesDivider
    class (or others) to determine the size of each axes. The unit
    classes define __call__ that returns a tuple of two floats,
    meaning relative and absolute sizes, respectively.

    Note that this class is nothing more than a simple tuple of two
    floats. Take a look at the Divider class to see how these two
    values are used.

    """

    class _Base(object):
        "Base class"
        pass

    class Fixed(_Base):
        "Simple fixed size  with relative part = 0"
        def __init__(self, fixed_size):
            self._fixed_size = fixed_size

        def get_size(self, renderer):
            rel_size = 0.
            abs_size = self._fixed_size
            return rel_size, abs_size

    class Scaled(_Base):
        "Simple scaled(?) size with absolute part = 0"
        def __init__(self, scalable_size):
            self._scalable_size = scalable_size

        def get_size(self, renderer):
            rel_size = self._scalable_size
            abs_size = 0.
            return rel_size, abs_size

    Scalable=Scaled

    class AxesX(_Base):
        """
        Scaled size whose relative part corresponds to the data width
        of the given axes
        """
        def __init__(self, axes, aspect=1.):
            self._axes = axes
            self._aspect = aspect

        def get_size(self, renderer):
            l1, l2 = self._axes.get_xlim()
            rel_size = abs(l2-l1)*self._aspect
            abs_size = 0.
            return rel_size, abs_size

    class AxesY(_Base):
        """
        Scaled size whose relative part corresponds to the data height
        of the given axes
        """
        def __init__(self, axes, aspect=1.):
            self._axes = axes
            self._aspect = aspect

        def get_size(self, renderer):
            l1, l2 = self._axes.get_ylim()
            rel_size = abs(l2-l1)*self._aspect
            abs_size = 0.
            return rel_size, abs_size


    class MaxExtent(_Base):
        """
        Size whose absolute part is the largest width (or height) of
        the given list of artists.
        """
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
        """
        An instance whose size is a fraction of the reference size.
          ex) s = Fraction(0.3, AxesX(ax))
        """
        def __init__(self, fraction, fraction_ref):
            self._fraction_ref = fraction_ref
            self._fraction = fraction

        def get_size(self, renderer):
            if self._fraction_ref is None:
                return self._fraction, 0.
            else:
                r, a = self._fraction_ref.get_size(renderer)
                rel_size = r*self._fraction
                abs_size = a*self._fraction
                return rel_size, abs_size

    @classmethod
    def from_any(self, size, fraction_ref=None):
        """
        Creates Fixed unit when the first argument is a float, or a
        Fraction unit if that is a string that ends with %. The second
        argument is only meaningful when Fraction unit is created.

          >>> a = Size.from_any(1.2) # => Size.Fixed(1.2)
          >>> Size.from_any("50%", a) # => Size.Fraction(0.5, a)

        """
        if cbook.is_numlike(size):
            return Size.Fixed(size)
        elif cbook.is_string_like(size):
            if size[-1] == "%":
                return Size.Fraction(float(size[:-1])/100., fraction_ref)

        raise ValueError("Unknown format")



    class Padded(_Base):
        """
        Return a instance where the absolute part of *size* is
        increase by the amount of *pad*.
        """
        def __init__(self, size, pad):
            self._size = size
            self._pad = pad

        def get_size(self, renderer):
            r, a = self._size.get_size(renderer)
            rel_size = r
            abs_size = a + self._pad
            return rel_size, abs_size




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

        if rs_sum != 0.:
            k = (total_size - as_sum) / rs_sum
            return k
        else:
            return 0.


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
        """

        nx, nx1 : Integers specifying the column-position of the
        cell. When nx1 is None, a single nx-th column is
        specified. Otherwise location of columns spanning between nx
        to nx1 (but excluding nx1-th column) is is specified.

        ny, ny1 : same as nx and nx1, but for row positions.
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
        return AxesLocator(self, nx, ny, nx1, ny1)



class AxesLocator(object):
    """
    A simple callable object, initiallized with AxesDivider class,
    returns the position and size of the given cell.
    """
    def __init__(self, axes_divider, nx, ny, nx1=None, ny1=None):
        """
        'axes_divider' : An instance of AxesDivider class.

        nx, nx1 : Integers specifying the column-position of the
        cell. When nx1 is None, a single nx-th column is
        specified. Otherwise location of columns spanning between nx
        to nx1 (but excluding nx1-th column) is is specified.

        ny, ny1 : same as nx and nx1, but for row positions.
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

        return self._axes_divider.locate(self._nx + _xrefindex, self._ny + _yrefindex,
                                         self._nx1 + _xrefindex, self._ny1 + _yrefindex,
                                         renderer)



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

        #axes_class = type(self._axes)
        ax = self._get_new_axes(**kwargs)
        #ax = axes_class(self._axes.get_figure(),
        #                self._axes.get_position(original=True),
        #                **kwargs)
        locator = self.new_locator(nx=len(self._horizontal)-1, ny=0)
        ax.set_axes_locator(locator)

        return ax

    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):

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

        ax = self._get_new_axes(**kwargs)
        #axes_class = type(self._axes)
        #ax = axes_class(self._axes.get_figure(),
        #                self._axes.get_position(original=True),
        #                **kwargs)
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

#if hasattr(maxes.Axes, "get_axes_locator"):
#    LocatableAxes = maxes.Axes
#else:

from mpl_toolkits.axes_grid.axislines import Axes
LocatableAxes = locatable_axes_factory(Axes)


def make_axes_locatable(axes):
    if not hasattr(axes, "set_axes_locator"):
        new_class = locatable_axes_factory(type(axes))
        axes.__class__ = new_class

    divider = AxesDivider(axes)
    locator = divider.new_locator(nx=0, ny=0)
    axes.set_axes_locator(locator)

    return divider


