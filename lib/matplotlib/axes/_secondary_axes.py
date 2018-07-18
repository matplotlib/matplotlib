import collections
import numpy as np
import numbers

import warnings

import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.scale as mscale

from matplotlib.axes._base import _AxesBase

from matplotlib.ticker import (
    AutoLocator,
    FixedLocator,
    NullLocator,
    NullFormatter,
    FuncFormatter,
    ScalarFormatter,
    AutoMinorLocator,
)

def _make_inset_locator(rect, trans, parent):
    """
    Helper function to locate inset axes, used in
    `.Axes.inset_axes_from_bounds`.

    A locator gets used in `Axes.set_aspect` to override the default
    locations...  It is a function that takes an axes object and
    a renderer and tells `set_aspect` where it is to be placed.

    Here *rect* is a rectangle [l, b, w, h] that specifies the
    location for the axes in the transform given by *trans* on the
    *parent*.
    """
    _rect = mtransforms.Bbox.from_bounds(*rect)
    _trans = trans
    _parent = parent

    def inset_locator(ax, renderer):
        bbox = _rect
        bb = mtransforms.TransformedBbox(bbox, _trans)
        tr = _parent.figure.transFigure.inverted()
        bb = mtransforms.TransformedBbox(bb, tr)
        return bb

    return inset_locator


def _parse_conversion(name, otherargs):
    print(otherargs)
    if name == 'inverted':
        if otherargs is None:
            otherargs = [1.]
        otherargs = np.atleast_1d(otherargs)
        return _InvertTransform(otherargs[0])
    elif name == 'power':
        otherargs = np.atleast_1d(otherargs)
        return _PowerTransform(a=otherargs[0], b=otherargs[1])
    elif name == 'linear':
        otherargs = np.asarray(otherargs)
        return _LinearTransform(slope=otherargs[0], offset=otherargs[1])
    else:
        raise ValueError(f'"{name}" not a possible conversion string')

class Secondary_Xaxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, location, conversion, otherargs=None, **kwargs):
        self._conversion = conversion
        self._parent = parent
        self._otherargs = otherargs

        super().__init__(self._parent.figure, [0, 1., 1, 0.0001], **kwargs)

        self.set_location(location)

        # styling:
        self.yaxis.set_major_locator(mticker.NullLocator())
        self.yaxis.set_ticks_position('none')
        self.spines['right'].set_visible(False)
        self.spines['left'].set_visible(False)
        if self._y > 0.5:
            self.set_axis_orientation('top')
        else:
            self.set_axis_orientation('bottom')
        self.set_conversion(conversion, self._otherargs)

    def set_axis_orientation(self, orient):
        """
        Set if axes spine and labels are drawn at top or bottom of the
        axes.

        Parameters
        ----------
        orient :: string
            either 'top' or 'bottom'

        """

        self.spines[orient].set_visible(True)
        if orient == 'top':
            self.spines['bottom'].set_visible(False)
        else:
            self.spines['top'].set_visible(False)

        self.xaxis.set_ticks_position(orient)
        self.xaxis.set_label_position(orient)

    def set_location(self, location):
        """
        Set the vertical location of the axes in parent-normalized
        co-ordinates.

        Parameters
        ----------
        location : string or scalar
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom', scalar can be a float indicating the relative position
            on the parent axes to put the new axes, 0 being the bottom, and
            1.0 being the top.
        """

        self._loc = location
        # This puts the rectangle into figure-relative coordinates.
        if isinstance(self._loc, str):
            if self._loc == 'top':
                y = 1.
            elif self._loc == 'bottom':
                y = 0.
            else:
                raise ValueError("location must be 'bottom', 'top', or a "
                                  "float, not '{}'.".format(self._loc))

        else:
            y = self._loc
        bounds = [0, y, 1., 1e-10]
        transform = self._parent.transAxes
        secondary_locator = _make_inset_locator(bounds,
                                                transform, self._parent)
        bb = secondary_locator(None, None)

        # this locator lets the axes move if in data coordinates.
        # it gets called in `ax.apply_aspect() (of all places)
        self.set_axes_locator(secondary_locator)
        self._y = y

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks with list of *ticks*

        Parameters
        ----------
        ticks : list
            List of x-axis tick locations.

        minor : bool, optional
            If ``False`` sets major ticks, if ``True`` sets minor ticks.
            Default is ``False``.
        """
        ret = self.xaxis.set_ticks(ticks, minor=minor)
        self.stale = True

        lims = self._parent.get_xlim()
        self.set_xlim(self._convert.transform(lims))
        return ret


    def set_conversion(self, conversion, otherargs=None):
        """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        conversion : float, two-tuple of floats, transform, or string
            transform between the parent xaxis values and the secondary xaxis
            values.  If a single floats, a linear transform with the
            float as the slope is used.  If a 2-tuple of floats, the first
            is the slope, and the second the offset.

            If a transform is supplied, then the transform must have an
            inverse.

            For convenience a few common transforms are provided by using
            a string:
              - 'linear': as above.  ``otherargs = (slope, offset)`` must
              be supplied.
              - 'inverted': a/x where ``otherargs = a`` can be supplied
              (defaults to 1)
              - 'power': b x^a  where ``otherargs = (a, b)`` must be
              supplied

        """

        # make the _convert function...
        if isinstance(conversion, mtransforms.Transform):
            self._convert = conversion
            self.set_xscale('arbitrary', transform=conversion.inverted())
        elif isinstance(conversion, str):
            self._convert = _parse_conversion(conversion, otherargs)
            self.set_xscale('arbitrary', transform=self._convert.inverted())
        else:
            # linear conversion with offset
            if isinstance(conversion, numbers.Number):
                conversion = np.asanyarray([conversion])
            if len(conversion) > 2:
                raise ValueError('secondary_axes conversion can be a '
                                 'float, two-tuple of float, a transform '
                                 'with an inverse, or a string.')
            elif len(conversion) < 2:
                conversion = np.array([conversion, 0.])
            conversion = _LinearTransform(slope=conversion[0],
                                          offset=conversion[1])
            self._convert = conversion
            # this will track log/non log so long as the user sets...
            self.set_xscale(self._parent.get_xscale())


    def draw(self, renderer=None, inframe=False):
        """
        Draw the secondary axes.

        Consults the parent axes for its xlimits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_conversion` (or *conversion*
        parameter when axes initialized.)

        """
        lims = self._parent.get_xlim()
        order = lims[0] < lims[1]
        lims = self._convert.transform(lims)
        neworder = lims[0] < lims[1]
        if neworder != order:
            # flip because the transform will take care of the flipping..
            lims = lims[::-1]
        self.set_xlim(lims)
        super().draw(renderer=renderer, inframe=inframe)

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the secondary x-axis.

        Parameters
        ----------
        xlabel : str
            The label text.

        labelpad : scalar, optional, default: None
            Spacing in points between the label and the x-axis.

        Other Parameters
        ----------------
        **kwargs : `.Text` properties
            `.Text` properties control the appearance of the label.

        See also
        --------
        text : for information on how override and the optional args work
        """
        if labelpad is not None:
            self.xaxis.labelpad = labelpad
        return self.xaxis.set_label_text(xlabel, fontdict, **kwargs)

    def set_color(self, color):
        """
        Change the color of the secondary axes and all decorators

        Parameters
        ----------
        color : Matplotlib color
        """

        self.tick_params(axis='x', colors=color)
        self.spines['bottom'].set_color(color)
        self.spines['top'].set_color(color)
        self.xaxis.label.set_color(color)

    def get_tightbbox(self, renderer, call_axes_locator=True):
        """
        Return the tight bounding box of the axes.
        The dimension of the Bbox in canvas coordinate.

        If *call_axes_locator* is *False*, it does not call the
        _axes_locator attribute, which is necessary to get the correct
        bounding box. ``call_axes_locator==False`` can be used if the
        caller is only intereted in the relative size of the tightbbox
        compared to the axes bbox.
        """

        bb = []

        if not self.get_visible():
            return None

        locator = self.get_axes_locator()
        if locator and call_axes_locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        bb_xaxis = self.xaxis.get_tightbbox(renderer)
        if bb_xaxis:
            bb.append(bb_xaxis)

        bb.append(self.get_window_extent(renderer))

        _bbox = mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

        return _bbox

    def set_aspect(self, *args, **kwargs):
        """
        """
        warnings.warn("Secondary axes can't set the aspect ratio")


class Secondary_Yaxis(_AxesBase):
    """
    Class to hold a Secondary_Yaxis.
    """

    def __init__(self, parent, location, conversion, **kwargs):
        self._conversion = conversion
        self._parent = parent
        self._x = None # set in set_location

        super().__init__(self._parent.figure, [1., 0., 0.00001, 1.], **kwargs)

        self.set_location(location)

        # styling:
        self.xaxis.set_major_locator(mticker.NullLocator())
        self.xaxis.set_ticks_position('none')
        self.spines['top'].set_visible(False)
        self.spines['bottom'].set_visible(False)
        if self._x > 0.5:
            self.set_axis_orientation('right')
        else:
            self.set_axis_orientation('left')
        self.set_conversion(conversion)

    def set_axis_orientation(self, orient):
        """
        Set if axes spine and labels are drawn at left or right of the
        axis.

        Parameters
        ----------
        orient :: string
            either 'left' or 'right'

        """

        self.spines[orient].set_visible(True)
        if orient == 'left':
            self.spines['right'].set_visible(False)
        else:
            self.spines['left'].set_visible(False)

        self.yaxis.set_ticks_position(orient)
        self.yaxis.set_label_position(orient)

    def set_location(self, location):
        """
        Set the horizontal location of the axes in parent-normalized
        co-ordinates.

        Parameters
        ----------
        location : string or scalar
            The position to put the secondary axis.  Strings can be 'left' or
            'right', or scalar can be a float indicating the relative position
            on the parent axes to put the new axes, 0 being the left, and
            1.0 being the right.
        """

        self._loc = location
        # This puts the rectangle into figure-relative coordinates.
        if isinstance(self._loc, str):
            if self._loc == 'left':
                x = 0.
            elif self._loc == 'right':
                x = 1.
            else:
                raise ValueError("location must be 'left', 'right', or a "
                                  "float, not '{}'.".format(self._loc))
        else:
            x = self._loc
        bounds = [x, 0, 1e-10, 1.]
        transform = self._parent.transAxes
        secondary_locator = _make_inset_locator(bounds,
                                                transform, self._parent)
        bb = secondary_locator(None, None)

        # this locator lets the axes move if in data coordinates.
        # it gets called in `ax.apply_aspect() (of all places)
        self.set_axes_locator(secondary_locator)
        self._x = x

    def set_conversion(self, conversion):
        """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        conversion : tuple of floats or function
            conversion between the parent xaxis values and the secondary xaxis
            values.  If a tuple of floats, the floats are polynomial
            co-efficients, with the first entry the highest exponent's
            co-efficient (i.e. [2, 3, 1] is the same as
            ``xnew = 2 x**2 + 3 * x + 1``, passed to `numpy.polyval`).
            If a function is specified it should accept a float as input and
            return a float as the result.
        """

        # make the _convert function...
        if callable(conversion):
            self._convert = conversion
        else:
            if isinstance(conversion, numbers.Number):
                conversion = np.asanyarray([conversion])
            shp = len(conversion)
            if shp < 2:
                conversion = np.array([conversion, 0.])
            self._convert = lambda x: np.polyval(conversion, x)

    def draw(self, renderer=None, inframe=False):
        """
        Draw the secondary axes.

        Consults the parent axes for its xlimits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_conversion` (or *conversion*
        parameter when axes initialized.)

        """
        lims = self._parent.get_xlim()
        self.set_ylim(self._convert(lims))
        super().draw(renderer=renderer, inframe=inframe)

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the secondary y-axis.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : scalar, optional, default: None
            Spacing in points between the label and the x-axis.

        Other Parameters
        ----------------
        **kwargs : `.Text` properties
            `.Text` properties control the appearance of the label.

        See also
        --------
        text : for information on how override and the optional args work
        """
        if labelpad is not None:
            self.yaxis.labelpad = labelpad
        return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)

    def set_color(self, color):
        """
        Change the color of the secondary axes and all decorators

        Parameters
        ----------
        color : Matplotlib color
        """

        self.tick_params(axis='y', colors=color)
        self.spines['left'].set_color(color)
        self.spines['right'].set_color(color)
        self.yaxis.label.set_color(color)

    def get_tightbbox(self, renderer, call_axes_locator=True):
        """
        Return the tight bounding box of the axes.
        The dimension of the Bbox in canvas coordinate.

        If *call_axes_locator* is *False*, it does not call the
        _axes_locator attribute, which is necessary to get the correct
        bounding box. ``call_axes_locator==False`` can be used if the
        caller is only intereted in the relative size of the tightbbox
        compared to the axes bbox.
        """

        bb = []

        if not self.get_visible():
            return None

        locator = self.get_axes_locator()
        if locator and call_axes_locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        bb_yaxis = self.yaxis.get_tightbbox(renderer)
        if bb_yaxis:
            bb.append(bb_yaxis)

        bb.append(self.get_window_extent(renderer))

        _bbox = mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

        return _bbox

    def set_aspect(self, *args, **kwargs):
        """
        """
        warnings.warn("Secondary axes can't set the aspect ratio")


class _LinearTransform(mtransforms.AffineBase):
    """
    Linear transform 1d
    """
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, slope, offset):
        mtransforms.AffineBase.__init__(self)
        self._slope = slope
        self._offset = offset

    def transform_affine(self, values):
        return np.asarray(values) * self._slope + self._offset

    def inverted(self):
        return _InvertedLinearTransform(self._slope, self._offset)


class _InverseLinearTransform(mtransforms.AffineBase):
    """
    Inverse linear transform 1d
    """
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, slope, offset):
        mtransforms.AffineBase.__init__(self)
        self._slope = slope
        self._offset = offset

    def transform_affine(self, values):
        return (np.asarray(values) - self._offset) / self._slope

    def inverted(self):
        return _LinearTransform(self._slope, self._offset)

def _mask_out_of_bounds(a):
    """
    Return a Numpy array where all values outside ]0, 1[ are
    replaced with NaNs. If all values are inside ]0, 1[, the original
    array is returned.
    """
    a = numpy.array(a, float)
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return numpy.where(mask, numpy.nan, a)
    return a

class _InvertTransform(mtransforms.Transform):
    """
    Return a/x
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, fac, out_of_bounds='mask'):
        mtransforms.Transform.__init__(self)
        self._fac = fac
        self.out_of_bounds = out_of_bounds
        if self.out_of_bounds == 'mask':
            self._handle_out_of_bounds = _mask_out_of_bounds
        elif self.out_of_bounds == 'clip':
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`out_of_bounds` muse be either 'mask' or 'clip'")

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self._fac / values
        return q

    def inverted(self):
        """ we are just our own inverse """
        return _InvertTransform(1 / self._fac)


class _PowerTransform(mtransforms.Transform):
    """
    Return b * x^a
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, a, b, out_of_bounds='mask'):
        mtransforms.Transform.__init__(self)
        self._a = a
        self._b = b
        self.out_of_bounds = out_of_bounds
        if self.out_of_bounds == 'mask':
            self._handle_out_of_bounds = _mask_out_of_bounds
        elif self.out_of_bounds == 'clip':
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`out_of_bounds` muse be either 'mask' or 'clip'")

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self._b * (values ** self._a)
            print('forward', values, q)
        return q

    def inverted(self):
        """ we are just our own inverse """
        return _InversePowerTransform(self._a, self._b)


class _InversePowerTransform(mtransforms.Transform):
    """
    Return b * x^a
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, a, b, out_of_bounds='mask'):
        mtransforms.Transform.__init__(self)
        self._a = a
        self._b = b
        self.out_of_bounds = out_of_bounds
        if self.out_of_bounds == 'mask':
            self._handle_out_of_bounds = _mask_out_of_bounds
        elif self.out_of_bounds == 'clip':
            self._handle_out_of_bounds = _clip_out_of_bounds
        else:
            raise ValueError("`out_of_bounds` must be either 'mask' or 'clip'")

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q =  (values / self._b) ** (1 / self._a)
            print(values, q)
        return q

    def inverted(self):
        """ we are just our own inverse """
        return _PowerTransform(self._a, self._b)


class ArbitraryScale(mscale.ScaleBase):

    name = 'arbitrary'

    def __init__(self, axis, transform=mtransforms.IdentityTransform()):
        """
        TODO
        """
        self._transform = transform

    def get_transform(self):
        """
        The transform for linear scaling is just the
        :class:`~matplotlib.transforms.IdentityTransform`.
        """
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults for
        linear scaling.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

mscale.register_scale(ArbitraryScale)
