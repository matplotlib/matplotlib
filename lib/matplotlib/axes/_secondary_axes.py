import numpy as np

import matplotlib.cbook as cbook
import matplotlib.docstring as docstring
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.axes._base import _AxesBase


def _make_secondary_locator(rect, parent):
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
    def secondary_locator(ax, renderer):
        # delay evaluating transform until draw time because the
        # parent transform may have changed (i.e. if window reesized)
        bb = mtransforms.TransformedBbox(_rect, parent.transAxes)
        tr = parent.figure.transFigure.inverted()
        bb = mtransforms.TransformedBbox(bb, tr)
        return bb

    return secondary_locator


class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation,
                  location, functions, **kwargs):
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """

        self._functions = functions
        self._parent = parent
        self._orientation = orientation
        self._ticks_set = False

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
        self._parentscale = self._axis.get_scale()
        # this gets positioned w/o constrained_layout so exclude:
        self._layoutbox = None
        self._poslayoutbox = None

        self.set_location(location)
        self.set_functions(functions)

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

        if self._pos < 0.5:
            # flip the location strings...
            self._locstrings = self._locstrings[::-1]
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the axes.

        Parameters
        ----------
        align : str
            either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
        if align in self._locstrings:
            if align == self._locstrings[1]:
                # need to change the orientation.
                self._locstrings = self._locstrings[::-1]
            elif align != self._locstrings[0]:
                raise ValueError('"{}" is not a valid axis orientation, '
                                 'not changing the orientation;'
                                 'choose "{}" or "{}""'.format(align,
                                 self._locstrings[0], self._locstrings[1]))
            self.spines[self._locstrings[0]].set_visible(True)
            self.spines[self._locstrings[1]].set_visible(False)
            self._axis.set_ticks_position(align)
            self._axis.set_label_position(align)

    def set_location(self, location):
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized co-ordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent axes to put the new axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).
        """

        # This puts the rectangle into figure-relative coordinates.
        if isinstance(location, str):
            if location in ['top', 'right']:
                self._pos = 1.
            elif location in ['bottom', 'left']:
                self._pos = 0.
            else:
                raise ValueError("location must be '{}', '{}', or a "
                                 "float, not '{}'".format(location,
                                 self._locstrings[0], self._locstrings[1]))
        else:
            self._pos = location
        self._loc = location

        if self._orientation == 'x':
            bounds = [0, self._pos, 1., 1e-10]
        else:
            bounds = [self._pos, 0, 1e-10, 1]

        secondary_locator = _make_secondary_locator(bounds, self._parent)

        # this locator lets the axes move in the parent axes coordinates.
        # so it never needs to know where the parent is explicitly in
        # figure co-ordinates.
        # it gets called in `ax.apply_aspect() (of all places)
        self.set_axes_locator(secondary_locator)

    def apply_aspect(self, position=None):
        # docstring inherited.
        self._set_lims()
        super().apply_aspect(position)

    @cbook._make_keyword_only("3.2", "minor")
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
        self._ticks_set = True
        return ret

    def set_functions(self, functions):
        """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """

        if self._orientation == 'x':
            set_scale = self.set_xscale
            parent_scale = self._parent.get_xscale()
        else:
            set_scale = self.set_yscale
            parent_scale = self._parent.get_yscale()
        # we need to use a modified scale so the scale can receive the
        # transform.  Only types supported are linear and log10 for now.
        # Probably possible to add other transforms as a todo...
        if parent_scale == 'log':
            defscale = 'functionlog'
        else:
            defscale = 'function'

        if (isinstance(functions, tuple) and len(functions) == 2 and
            callable(functions[0]) and callable(functions[1])):
            # make an arbitrary convert from a two-tuple of functions
            # forward and inverse.
            self._functions = functions
        elif functions is None:
            self._functions = (lambda x: x, lambda x: x)
        else:
            raise ValueError('functions argument of secondary axes '
                             'must be a two-tuple of callable functions '
                             'with the first function being the transform '
                             'and the second being the inverse')
        # need to invert the roles here for the ticks to line up.
        set_scale(defscale, functions=self._functions[::-1])

    def draw(self, renderer=None, inframe=False):
        """
        Draw the secondary axes.

        Consults the parent axes for its limits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_functions` (or *functions*
        parameter when axes initialized.)
        """
        self._set_lims()
        # this sets the scale in case the parent has set its scale.
        self._set_scale()
        super().draw(renderer=renderer, inframe=inframe)

    def _set_scale(self):
        """
        Check if parent has set its scale
        """

        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        if self._orientation == 'y':
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        if pscale == self._parentscale:
            return
        else:
            self._parentscale = pscale

        if pscale == 'log':
            defscale = 'functionlog'
        else:
            defscale = 'function'

        if self._ticks_set:
            ticks = self._axis.get_ticklocs()

        # need to invert the roles here for the ticks to line up.
        set_scale(defscale, functions=self._functions[::-1])

        # OK, set_scale sets the locators, but if we've called
        # axsecond.set_ticks, we want to keep those.
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))

    def _set_lims(self):
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary axes.
        """
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        if self._orientation == 'y':
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        if neworder != order:
            # Flip because the transform will take care of the flipping.
            lims = lims[::-1]
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        """
        Secondary axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
        cbook._warn_external("Secondary axes can't set the aspect ratio")

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
        Change the color of the secondary axes and all decorators.

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


_secax_docstring = '''
Warnings
--------
This method is experimental as of 3.1, and the API may change.

Parameters
----------
location : {'top', 'bottom', 'left', 'right'} or float
    The position to put the secondary axis.  Strings can be 'top' or
    'bottom' for orientation='x' and 'right' or 'left' for
    orientation='y'. A float indicates the relative position on the
    parent axes to put the new axes, 0.0 being the bottom (or left)
    and 1.0 being the top (or right).

functions : 2-tuple of func, or Transform with an inverse

    If a 2-tuple of functions, the user specifies the transform
    function and its inverse.  i.e.
    `functions=(lambda x: 2 / x, lambda x: 2 / x)` would be an
    reciprocal transform with a factor of 2.

    The user can also directly supply a subclass of
    `.transforms.Transform` so long as it has an inverse.

    See :doc:`/gallery/subplots_axes_and_figures/secondary_axis`
    for examples of making these conversions.


Other Parameters
----------------
**kwargs : `~matplotlib.axes.Axes` properties.
    Other miscellaneous axes parameters.

Returns
-------
ax : axes._secondary_axes.SecondaryAxis
'''
docstring.interpd.update(_secax_docstring=_secax_docstring)
