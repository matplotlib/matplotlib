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


def _make_secondary_locator(rect, trans, parent):
    """
    Helper function to locate the secondary axes.

    A locator gets used in `Axes.set_aspect` to override the default
    locations...  It is a function that takes an axes object and
    a renderer and tells `set_aspect` where it is to be placed.

    This locator make the transform be in axes-relative co-coordinates
    because that is how we specify the "location" of the secondary axes.

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
        raise ValueError('"{}" not a possible conversion string'.format(name))


class Secondary_Axis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation,
                  location, conversion, otherargs=None, **kwargs):

        self._conversion = conversion
        self._parent = parent
        self._otherargs = otherargs
        self._orientation = orientation

        if self._orientation == 'x':
            super().__init__(self._parent.figure, [0, 1., 1, 0.0001], **kwargs)
            self._axis = self.xaxis
            self._locstrings = ['top', 'bottom']
            self._otherstrings = ['left', 'right']
        elif self._orientation == 'y':
            super().__init__(self._parent.figure, [0, 1., 0.0001, 1], **kwargs)
            self._axis = self.yaxis
            self._locstrings = ['right', 'left']
            self._otherstrings = ['top', 'bottom']

        self.set_location(location)
        self.set_conversion(conversion, self._otherargs)

        # styling:
        if self._orientation == 'x':
            otheraxis = self.yaxis
        else:
            otheraxis = self.xaxis

        otheraxis.set_major_locator(mticker.NullLocator())
        otheraxis.set_ticks_position('none')

        for st in self._otherstrings:
            self.spines[st].set_visible(False)
        for st in self._locstrings:
            self.spines[st].set_visible(True)

        if  self._pos < 0.5:
            # flip the location strings...
            self._locstrings = self._locstrings[::-1]
        self.set_axis_orientation(self._locstrings[0])

    def set_axis_orientation(self, orient):
        """
        Set if axes spine and labels are drawn at top or bottom of the
        axes.

        Parameters
        ----------
        orient :: string
            either 'top' or 'bottom'

        """
        if orient in self._locstrings:
            if orient == self._locstrings[1]:
                # need to change the orientation.
                self._locstrings = self._locstrings[::-1]
            elif orient != self._locstrings[0]:
                warnings.warn('"{}" is not a valid axis orientation, '
                            'not changing the orientation;'
                            'choose "{}" or "{}""'.format(orient,
                            self._locstrings[0], self._locstrings[1]))
            self.spines[self._locstrings[0]].set_visible(True)
            self.spines[self._locstrings[1]].set_visible(False)
            self._axis.set_ticks_position(orient)
            self._axis.set_label_position(orient)

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

        # This puts the rectangle into figure-relative coordinates.
        if isinstance(location, str):
            if location in ['top', 'right']:
                self._pos = 1.
            elif location in ['bottom', 'left']:
                self._pos = 0.
            else:
                warnings.warn("location must be '{}', '{}', or a "
                                  "float, not '{}'.".format(location,
                                  self._locstrings[0], self._locstrings[1]))
                return
        else:
            self._pos = location
        self._loc = location

        if self._orientation == 'x':
            bounds = [0, self._pos, 1., 1e-10]
        else:
            bounds = [self._pos, 0, 1e-10, 1]

        transform = self._parent.transAxes
        secondary_locator = _make_secondary_locator(bounds,
                                                transform, self._parent)
        bb = secondary_locator(None, None)

        # this locator lets the axes move in the parent axes coordinates.
        # so it never needs to know where the parent is explicitly in
        # figure co-ordinates.
        # it gets called in `ax.apply_aspect() (of all places)
        self.set_axes_locator(secondary_locator)

    def set_ticks(self, ticks, minor=False):
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
        ret = self._axis.set_ticks(ticks, minor=minor)
        self.stale = True

        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            self.set_xlim(self._convert.transform(lims))
        else:
            lims = self._parent.get_ylim()
            self.set_ylim(self._convert.transform(lims))

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

        if self._orientation == 'x':
            set_scale = self.set_xscale
        else:
            set_scale = self.set_yscale

        # make the _convert function...
        if isinstance(conversion, mtransforms.Transform):
            self._convert = conversion
            set_scale('arbitrary', transform=conversion.inverted())
        elif isinstance(conversion, str):
            self._convert = _parse_conversion(conversion, otherargs)
            set_scale('arbitrary', transform=self._convert.inverted())
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
            set_scale(self._parent.get_xscale())

    def draw(self, renderer=None, inframe=False):
        """
        Draw the secondary axes.

        Consults the parent axes for its xlimits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_conversion` (or *conversion*
        parameter when axes initialized.)

        """
        # check parent scale...  Make these match....
        if self._orientation == 'x':
            scale = self._parent.get_xscale()
            self.set_xscale(scale)
        if self._orientation == 'y':
            scale = self._parent.get_yscale()
            self.set_yscale(scale)

        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        if self._orientation == 'y':
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        print('parent', lims)
        order = lims[0] < lims[1]
        lims = self._convert.transform(lims)
        neworder = lims[0] < lims[1]
        if neworder != order:
            # flip because the transform will take care of the flipping..
            # lims = lims[::-1]
            pass
        print('childs', lims)

        set_lim(lims)
        super().draw(renderer=renderer, inframe=inframe)

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
        if self._orientation == 'x':
            bb_axis = self.xaxis.get_tightbbox(renderer)
        else:
            bb_axis = self.yaxis.get_tightbbox(renderer)
        if bb_axis:
            bb.append(bb_axis)

        bb.append(self.get_window_extent(renderer))

        _bbox = mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

        return _bbox

    def set_aspect(self, *args, **kwargs):
        """
        """
        warnings.warn("Secondary axes can't set the aspect ratio")

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the x-axis.

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

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the x-axis.

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

        if self._orientation == 'x':
            self.tick_params(axis='x', colors=color)
            self.spines['bottom'].set_color(color)
            self.spines['top'].set_color(color)
            self.xaxis.label.set_color(color)
        else:
            self.tick_params(axis='y', colors=color)
            self.spines['left'].set_color(color)
            self.spines['right'].set_color(color)
            self.yaxis.label.set_color(color)


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
        return _InverseLinearTransform(self._slope, self._offset)


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
    a = np.array(a, float)
    mask = (a <= 0.0) | (a >= 1.0)
    if mask.any():
        return np.where(mask, np.nan, a)
    return a


class _InvertTransform(mtransforms.Transform):
    """
    Return a/x
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, fac):
        mtransforms.Transform.__init__(self)
        self._fac = fac

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self._fac / values
        print('q', values, q)
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

    def __init__(self, a, b):
        mtransforms.Transform.__init__(self)
        self._a = a
        self._b = b

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self._b * (values ** self._a)
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

    def __init__(self, a, b):
        mtransforms.Transform.__init__(self)
        self._a = a
        self._b = b

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (values / self._b) ** (1 / self._a)
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
        print('tranform', self._transform)
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
