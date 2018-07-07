import collections
import numpy as np
import numbers

import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

from matplotlib.axes._base import _AxesBase


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


class Secondary_Xaxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, location, conversion, **kwargs):
        self._conversion = conversion
        self._parent = parent

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
        self.set_conversion(conversion)

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
        self.set_xlim(self._convert(lims))
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
