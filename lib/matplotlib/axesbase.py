
# We need this future import because we unfortunately have
# a module name collision with the standard module called "collections".
from __future__ import absolute_import
from collections import defaultdict

import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib import cbook

class AxesBase(martist.Artist):
    """
    This class serves to logically segregate some of the methods of the
    :class:`~matplotlib.axes.Axes` class into their own base class.
    These methods are largely generalized maintenance operations and
    other book-keeping actions.  Ultimately, the objective is for the
    Axes class will contain only pure plotting methods.

    It is still intended for developers and extension writers to continue
    subclassing the Axes class, and no existing code will be broken by
    this migration effort.

    """

    _shared_axes = defaultdict(cbook.Grouper)

    def __init__(self, fig, rect, axis_names,
                 axisbg=None, # defaults to rc axes.facecolor
                 frameon=True,
                 label='',
                 share=None,
                 scale=None):
        martist.Artist.__init__(self)
        self._position = (rect if isinstance(rect, mtransforms.Bbox) else
                          mtransforms.Bbox.from_bounds(*rect))

        self._axis_names = axis_names
        self._share = share or {name:None for name in axis_names}   
        self._scale = scale or {name:None for name in axis_names}

        self._originalPosition = self._position.frozen()
        self.set_axes(self)
        self.set_aspect('auto')
        self._adjustable = 'box'
        self.set_anchor('C')

        # TODO: Backwards compatibility shim
        self._sharex = self._share.get('x', None)
        self._sharey = self._share.get('y', None)

        for axis_name in axis_names :
            shared = self._share.get(axis_name, None)
            if shared is not None:
                self._shared_axes[axis_name].join(self, shared)
                # 'box' and 'datalim' are equivalent now.
                if shared._adjustable == 'box':
                    shared._adjustable = 'datalim'
                self._adjustable = 'datalim'

        self.set_label(label)
        self.set_figure(fig)

    def ishold(self):
        """return the HOLD status of the axes"""
        return self._hold

    def hold(self, b=None):
        """
        Call signature::

          hold(b=None)

        Set the hold state.  If *hold* is *None* (default), toggle the
        *hold* state.  Else set the *hold* state to boolean value *b*.

        Examples::

          # toggle hold
          hold()

          # turn hold on
          hold(True)

          # turn hold off
          hold(False)


        When hold is *True*, subsequent plot commands will be added to
        the current axes.  When hold is *False*, the current axes and
        figure will be cleared on the next plot command

        """
        if b is None:
            self._hold = not self._hold
        else:
            self._hold = b

    def get_aspect(self):
        return self._aspect

    def set_aspect(self, aspect, adjustable=None, anchor=None):
        """
        *aspect*

          ========   ================================================
          value      description
          ========   ================================================
          'auto'     automatic; fill position rectangle with data
          'normal'   same as 'auto'; deprecated
          'equal'    same scaling from data to plot units for x and y
           num       a circle will be stretched such that the height
                     is num times the width. aspect=1 is the same as
                     aspect='equal'.
          ========   ================================================

        *adjustable*

          ============   =====================================
          value          description
          ============   =====================================
          'box'          change physical size of axes
          'datalim'      change xlim or ylim
          'box-forced'   same as 'box', but axes can be shared
          ============   =====================================

        'box' does not allow axes sharing, as this can cause
        unintended side effect. For cases when sharing axes is
        fine, use 'box-forced'.

        *anchor*

          =====   =====================
          value   description
          =====   =====================
          'C'     centered
          'SW'    lower left corner
          'S'     middle of bottom edge
          'SE'    lower right corner
          etc.
          =====   =====================

        """
        if aspect in ('normal', 'auto'):
            self._aspect = 'auto'
        elif aspect == 'equal':
            self._aspect = 'equal'
        else:
            self._aspect = float(aspect) # raise ValueError if necessary

        if adjustable is not None:
            self.set_adjustable(adjustable)
        if anchor is not None:
            self.set_anchor(anchor)

    def get_adjustable(self):
        return self._adjustable

    def set_adjustable(self, adjustable):
        """
        ACCEPTS: [ 'box' | 'datalim' | 'box-forced']
        """
        if adjustable in ('box', 'datalim', 'box-forced'):
            if any((self in shared) for shared in self._shared_axes.values()):
                if adjustable == 'box':
                    raise ValueError(
                        'adjustable must be "datalim" for shared axes')
            self._adjustable = adjustable
        else:
            raise ValueError('argument must be "box", or "datalim"')

    def get_anchor(self):
        return self._anchor

    def set_anchor(self, anchor):
        """
        *anchor*

          =====  ============
          Value  Description
          =====  ============
          'C'    center
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

    def set_figure(self, fig):
        """
        Set the :class:`~matplotlib.axes.Axes`'s figure.

        Accepts a :class:`~matplotlib.figure.Figure` instance.

        """
        martist.Artist.set_figure(self, fig)

        self.bbox = mtransforms.TransformedBbox(self._position, fig.transFigure)
        #these will be updated later as data is added
        self.dataLim = mtransforms.Bbox.unit()
        self.viewLim = mtransforms.Bbox.unit()
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        self._set_lim_and_transforms()

    def _set_lim_and_transforms(self):
        """
        set the *dataLim* and *viewLim*
        :class:`~matplotlib.transforms.Bbox` attributes and the
        *transScale*, *transData*, *transLimits* and *transAxes*
        transformations.

        .. note::

            This method is primarily used by rectilinear projections
            of the :class:`~matplotlib.axes.Axes` class, and is meant
            to be overridden by new kinds of projection axes that need
            different transformations and limits. (See
            :class:`~matplotlib.projections.polar.PolarAxes` for an
            example).

        """
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # Transforms the x and y axis separately by a scale factor.
        # It is assumed that this part will have non-linear components
        # (e.g. for a log scale).
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewLim, self.transScale))

        # The parentheses are important for efficiency here -- they
        # group the last two (which are usually affines) separately
        # from the first (which, with log-scaling can be non-affine).
        self.transData = self.transScale + (self.transLimits + self.transAxes)

        self._xaxis_transform = mtransforms.blended_transform_factory(
                self.transData, self.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
                self.transAxes, self.transData)

    def _get_axis_transform(self, which, axis_trans, tick1_spine, tick2_spine):
        """
        Get the transformation used for drawing axis labels, ticks
        and gridlines.  This is meant as an internal method for the migration
        process.

        Values for "which": 'grid', 'tick1', 'tick2'
        """
        if which=='grid':
            return axis_trans
        elif which=='tick1':
            return tick1_spine.get_spine_transform()
        elif which=='tick2':
            return tick2_spine.get_spine_transform()
        else:
            raise ValueError('unknown value for "which"')

    def get_position(self, original=False):
        'Return the a copy of the axes rectangle as a Bbox'
        if original:
            return self._originalPosition.frozen()
        else:
            return self._position.frozen()


    def set_position(self, pos, which='both'):
        """
        Set the axes position with::

          pos = [left, bottom, width, height]

        in relative 0,1 coords, or *pos* can be a
        :class:`~matplotlib.transforms.Bbox`

        There are two position variables: one which is ultimately
        used, but which may be modified by :meth:`apply_aspect`, and a
        second which is the starting point for :meth:`apply_aspect`.


        Optional keyword arguments:
          *which*

            ==========   ====================
            value        description
            ==========   ====================
            'active'     to change the first
            'original'   to change the second
            'both'       to change both
            ==========   ====================

        """
        if not isinstance(pos, mtransforms.BboxBase):
            pos = mtransforms.Bbox.from_bounds(*pos)
        if which in ('both', 'active'):
            self._position.set(pos)
        if which in ('both', 'original'):
            self._originalPosition.set(pos)

    def reset_position(self):
        """Make the original position the active position"""
        pos = self.get_position(original=True)
        self.set_position(pos, which='active')

    def set_axes_locator(self, locator):
        """
        Set axes_locator

        ACCEPT : a callable object which takes an axes instance and
                 renderer and returns a bbox.
        """
        self._axes_locator = locator

    def get_axes_locator(self):
        """
        Return axes_locator
        """
        return self._axes_locator

