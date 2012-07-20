
# We need this future import because we unfortunately have
# a module name collision with the standard module called "collections".
from __future__ import absolute_import
from collections import defaultdict, OrderedDict

import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib import cbook
from matplotlib import docstring
from matplotlib import rcParams

is_string_like = cbook.is_string_like

def _string_to_bool(s):
    if not is_string_like(s):
        return s
    if s == 'on':
        return True
    if s == 'off':
        return False
    raise ValueError("string argument must be either 'on' or 'off'")




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
    _axis_classes = {}
    _axis_locs = {}

    def __init__(self, fig, rect, axis_names,
                 axisbg=None, # defaults to rc axes.facecolor
                 frameon=True,
                 label='',
                 share=None,
                 scale=None,
                 axes_locator=None):
        martist.Artist.__init__(self)
        self._position = (rect if isinstance(rect, mtransforms.Bbox) else
                          mtransforms.Bbox.from_bounds(*rect))

        self._share = share or dict([(name, None) for name in axis_names])
        self._scale = scale or dict([(name, None) for name in axis_names])
        self._axis_transforms = dict([(name, None) for name in axis_names])
        self._axis_objs = OrderedDict([(name, None) for name in axis_names])

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
        self.set_axes_locator(axes_locator)

        self.spines = self._gen_axes_spines()

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
        raise NotImplementedError("The %s class needs to define"
                                  " _set_lim_and_transforms")

    def _get_axis_transform(self, which, axis_name):
        """
        Get the transformation used for drawing axis labels, ticks
        and gridlines.  This is meant as an internal method for the migration
        process.

        Values for "which": 'grid', 'tick1', 'tick2'
        """
        spine_locs = self._axis_locs[axis_name]
        if which=='grid':
            return self._axis_transforms[axis_name]
        elif which=='tick1':
            return self.spines[spine_locs[0]].get_spine_transform()
        elif which=='tick2':
            return self.spines[spine_locs[1]].get_spine_transform()
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

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        """
        Returns a dict whose keys are spine names and values are
        Line2D or Patch instances. Each element is used to draw a
        spine of the axes.

        In the standard axes, this is a single line segment, but in
        other projections it may not be.

        .. note::

            Intended to be overridden by new projection types.

        """
        raise NotImplementedError("The subclass %s must define"
                                  " _gen_axes_spines()" % type(self))
    def _init_axis(self):
        try:
            for name in self._axis_objs:
                self._axis_objs[name] = self._axis_classes[name](self)
                for loc in self._axis_locs[name]:
                    self.spines[loc].register_axis(self._axis_objs[name])
        except KeyError:
            # TODO: Add information to this exception to make it clearer what
            # is wrong.
            raise

        self._update_transScale()

    def _update_transScale(self):
        self.transScale.set(
            mtransforms.blended_transform_factory(
                *[self._axis_objs[name].get_transform() for name in
                  self._axis_objs]))
        if hasattr(self, "lines"):
            for line in self.lines:
                try:
                    line._transformed_path.invalidate()
                except AttributeError:
                    pass

    def _set_artist_props(self, a):
        """set the boilerplate props for artists added to axes"""
        a.set_figure(self.figure)
        if not a.is_transform_set():
            a.set_transform(self.transData)

        a.set_axes(self)

    def get_legend(self):
        """Return the legend.Legend instance, or None if no legend is defined"""
        return self.legend_

    def get_images(self):
        """return a list of Axes images contained by the Axes"""
        return cbook.silent_list('AxesImage', self.images)

    def get_lines(self):
        """Return a list of lines contained by the Axes"""
        return cbook.silent_list('Line2D', self.lines)

    #### Adding and tracking artists

    def _sci(self, im):
        """
        helper for :func:`~matplotlib.pyplot.sci`;
        do not use elsewhere.
        """
        if isinstance(im, mcontour.ContourSet):
            if im.collections[0] not in self.collections:
                raise ValueError(
                    "ContourSet must be in current Axes")
        elif im not in self.images and im not in self.collections:
            raise ValueError(
            "Argument must be an image, collection, or ContourSet in this Axes")
        self._current_image = im

    def _gci(self):
        """
        Helper for :func:`~matplotlib.pyplot.gci`;
        do not use elsewhere.
        """
        return self._current_image

    def has_data(self):
        """
        Return *True* if any artists have been added to axes.

        This should not be used to determine whether the *dataLim*
        need to be updated, and may not actually be useful for
        anything.
        """
        return (
            len(self.collections) +
            len(self.images) +
            len(self.lines) +
            len(self.patches))>0

    def add_artist(self, a):
        """
        Add any :class:`~matplotlib.artist.Artist` to the axes.

        Returns the artist.
        """
        a.set_axes(self)
        self.artists.append(a)
        self._set_artist_props(a)
        a.set_clip_path(self.patch)
        a._remove_method = lambda h: self.artists.remove(h)
        return a

    def add_collection(self, collection, autolim=True):
        """
        Add a :class:`~matplotlib.collections.Collection` instance
        to the axes.

        Returns the collection.
        """
        label = collection.get_label()
        if not label:
            collection.set_label('_collection%d'%len(self.collections))
        self.collections.append(collection)
        self._set_artist_props(collection)

        if collection.get_clip_path() is None:
            collection.set_clip_path(self.patch)
        if autolim:
            if collection._paths and len(collection._paths):
                self.update_datalim(collection.get_datalim(self.transData))

        collection._remove_method = lambda h: self.collections.remove(h)
        return collection

    def add_line(self, line):
        """
        Add a :class:`~matplotlib.lines.Line2D` to the list of plot
        lines

        Returns the line.
        """
        self._set_artist_props(line)
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)

        self._update_line_limits(line)
        if not line.get_label():
            line.set_label('_line%d' % len(self.lines))
        self.lines.append(line)
        line._remove_method = lambda h: self.lines.remove(h)
        return line

    def add_patch(self, p):
        """
        Add a :class:`~matplotlib.patches.Patch` *p* to the list of
        axes patches; the clipbox will be set to the Axes clipping
        box.  If the transform is not set, it will be set to
        :attr:`transData`.

        Returns the patch.
        """

        self._set_artist_props(p)
        if p.get_clip_path() is None:
            p.set_clip_path(self.patch)
        self._update_patch_limits(p)
        self.patches.append(p)
        p._remove_method = lambda h: self.patches.remove(h)
        return p

    def add_table(self, tab):
        """
        Add a :class:`~matplotlib.tables.Table` instance to the
        list of axes tables

        Returns the table.
        """
        self._set_artist_props(tab)
        self.tables.append(tab)
        tab.set_clip_path(self.patch)
        tab._remove_method = lambda h: self.tables.remove(h)
        return tab

    def add_container(self, container):
        """
        Add a :class:`~matplotlib.container.Container` instance
        to the axes.

        Returns the collection.
        """
        label = container.get_label()
        if not label:
            container.set_label('_container%d'%len(self.containers))
        self.containers.append(container)
        container.set_remove_method(lambda h: self.containers.remove(container))
        return container

    def relim(self):
        """
        Recompute the data limits based on current artists.

        At present, :class:`~matplotlib.collections.Collection`
        instances are not supported.

        """
        # Collections are deliberately not supported (yet); see
        # the TODO note in artists.py.
        self.dataLim.ignore(True)
        self.ignore_existing_data_limits = True
        for line in self.lines:
            self._update_line_limits(line)

        for p in self.patches:
            self._update_patch_limits(p)

    def update_datalim_bounds(self, bounds):
        """
        Update the datalim to include the given
        :class:`~matplotlib.transforms.Bbox` *bounds*
        """
        self.dataLim.set(mtransforms.Bbox.union([self.dataLim, bounds]))

    def in_axes(self, mouseevent):
        """
        Return *True* if the given *mouseevent* (in display coords)
        is in the Axes
        """
        return self.patch.contains(mouseevent)[0]

    def set_rasterization_zorder(self, z):
        """
        Set zorder value below which artists will be rasterized
        """
        self._rasterization_zorder = z

    def get_rasterization_zorder(self):
        """
        Get zorder value below which artists will be rasterized
        """
        return self._rasterization_zorder

    def draw_artist(self, a):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)

        """
        assert self._cachedRenderer is not None
        a.draw(self._cachedRenderer)

    def redraw_in_frame(self):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)

        """
        assert self._cachedRenderer is not None
        self.draw(self._cachedRenderer, inframe=True)

    def get_renderer_cache(self):
        return self._cachedRenderer

    #### Axes region characteristics

    def get_frame_on(self):
        """
        Get whether the axes frame patch is drawn
        """
        return self._frameon

    def set_frame_on(self, b):
        """
        Set whether the axes frame patch is drawn

        ACCEPTS: [ *True* | *False* ]

        """
        self._frameon = b

    def get_axisbelow(self):
        """
        Get whether axis below is true or not
        """
        return self._axisbelow

    def set_axisbelow(self, b):
        """
        Set whether the axis ticks and gridlines are above or below most artists

        ACCEPTS: [ *True* | *False* ]

        """
        self._axisbelow = b

    @docstring.dedent_interpd
    def grid(self, b=None, which='major', axis='both', **kwargs):
        """
        Call signature::

           grid(self, b=None, which='major', axis='both', **kwargs)

        Set the axes grids on or off; *b* is a boolean.  (For MATLAB
        compatibility, *b* may also be a string, 'on' or 'off'.)

        If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
        *kwargs* are supplied, it is assumed that you want a grid and *b*
        is thus set to *True*.

        *which* can be 'major' (default), 'minor', or 'both' to control
        whether major tick grids, minor tick grids, or both are affected.

        *axis* can be 'both' (default), or the name of the axis (such as
        'x' or 'y') to control which set of gridlines are drawn.

        *kwargs* are used to set the grid line properties, eg::

           ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

        %(Line2D)s

        """
        if len(kwargs):
            b = True
        b = _string_to_bool(b)

        for name, axis_obj in self._axis_objs.iteritems():
            if axis == name or axis == 'both':
                axis_obj.grid(b, which=which, **kwargs)

    def ticklabel_format(self, **kwargs):
        """
        Convenience method for manipulating the ScalarFormatter
        used by default for linear axes.

        Optional keyword arguments:

          ============   =========================================
          Keyword        Description
          ============   =========================================
          *style*        [ 'sci' (or 'scientific') | 'plain' ]
                         plain turns off scientific notation
          *scilimits*    (m, n), pair of integers; if *style*
                         is 'sci', scientific notation will
                         be used for numbers outside the range
                         10`-m`:sup: to 10`n`:sup:.
                         Use (0,0) to include all numbers.
          *useOffset*    [True | False | offset]; if True,
                         the offset will be calculated as needed;
                         if False, no offset will be used; if a
                         numeric offset is specified, it will be
                         used.
          *axis*         [ 'both' | or name of axis (i.e., 'x' or 'y']
                         'both' means all axis in the case of 3d plots.
          *useLocale*    If True, format the number according to
                         the current locale.  This affects things
                         such as the character used for the
                         decimal separator.  If False, use
                         C-style (English) formatting.  The
                         default setting is controlled by the
                         axes.formatter.use_locale rcparam.
          ============   =========================================

        Only the major ticks are affected.
        If the method is called when the
        :class:`~matplotlib.ticker.ScalarFormatter` is not the
        :class:`~matplotlib.ticker.Formatter` being used, an
        :exc:`AttributeError` will be raised.

        """
        style = kwargs.pop('style', '').lower()
        scilimits = kwargs.pop('scilimits', None)
        useOffset = kwargs.pop('useOffset', None)
        useLocale = kwargs.pop('useLocale', None)
        axis = kwargs.pop('axis', 'both').lower()
        if scilimits is not None:
            try:
                m, n = scilimits
                m+n+1  # check that both are numbers
            except (ValueError, TypeError):
                raise ValueError("scilimits must be a sequence of 2 integers")
        if style[:3] == 'sci':
            sb = True
        elif style in ['plain', 'comma']:
            sb = False
            if style == 'plain':
                cb = False
            else:
                cb = True
                raise NotImplementedError("comma style remains to be added")
        elif style == '':
            sb = None
        else:
            raise ValueError("%s is not a valid style value" % style)

        try:
            frmttrs = [axis_obj.major.formatter for name, axis_obj in
                        self._axis_objs.iteritems() if
                        (axis == 'both' or axis == name)]
            if sb is not None:
                for formattrs in frmttrs:
                    formattrs.set_scientific(sb)
            if scilimits is not None:
                for formattrs in frmttrs:
                    formattrs.set_powerlimits(scilimits)
            if useOffset is not None:
                for formattrs in frmttrs:
                    formattrs.set_useOffset(useOffset)
            if useLocale is not None:
                for formattrs in frmttrs:
                    formattrs.set_useLocale(useLocale)
        except AttributeError:
            raise AttributeError(
                "This method only works with the ScalarFormatter.")

    def set_axis_off(self):
        """turn off the axis"""
        self.axison = False

    def set_axis_on(self):
        """turn on the axis"""
        self.axison = True

    def get_axis_bgcolor(self):
        """Return the axis background color"""
        return self._axisbg

    def set_axis_bgcolor(self, color):
        """
        set the axes background color

        ACCEPTS: any matplotlib color - see
        :func:`~matplotlib.pyplot.colors`

        """

        self._axisbg = color
        self.patch.set_facecolor(color)

    #### Interactive manipulation

    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.
        """
        return True

    def can_pan(self) :
        """
        Return *True* if this axes supports any pan/zoom button functionality.
        """
        return True

    def get_navigate(self):
        """
        Get whether the axes responds to navigation commands
        """
        return self._navigate

    def set_navigate(self, b):
        """
        Set whether the axes responds to navigation toolbar commands

        ACCEPTS: [ *True* | *False* ]
        """
        self._navigate = b

    def get_navigate_mode(self):
        """
        Get the navigation toolbar button status: 'PAN', 'ZOOM', or None
        """
        return self._navigate_mode

    def set_navigate_mode(self, b):
        """
        Set the navigation toolbar button status;

        .. warning::
            this is not a user-API function.

        """
        self._navigate_mode = b

    def start_pan(self, x, y, button):
        """
        Called when a pan operation has started.

        *x*, *y* are the mouse coordinates in display coords.
        button is the mouse button number:

        * 1: LEFT
        * 2: MIDDLE
        * 3: RIGHT

        .. note::

            Intended to be overridden by new projection types.

        """
        self._pan_start = cbook.Bunch(
            lim           = self.viewLim.frozen(),
            trans         = self.transData.frozen(),
            trans_inverse = self.transData.inverted().frozen(),
            bbox          = self.bbox.frozen(),
            x             = x,
            y             = y
            )

    def end_pan(self):
        """
        Called when a pan operation completes (when the mouse button
        is up.)

        .. note::

            Intended to be overridden by new projection types.

        """
        del self._pan_start

    def get_cursor_props(self):
        """
        Return the cursor propertiess as a (*linewidth*, *color*)
        tuple, where *linewidth* is a float and *color* is an RGBA
        tuple
        """
        return self._cursorProps

    def set_cursor_props(self, *args):
        """
        Set the cursor property as::

          ax.set_cursor_props(linewidth, color)

        or::

          ax.set_cursor_props((linewidth, color))

        ACCEPTS: a (*float*, *color*) tuple
        """
        if len(args) == 1:
            lw, c = args[0]
        elif len(args) == 2:
            lw, c = args
        else:
            raise ValueError('args must be a (linewidth, color) tuple')

        c = mcolors.colorConverter.to_rgba(c)
        self._cursorProps = (lw, c)

    def get_children(self):
        """return a list of child artists"""
        children = []
        children.extend(self._axis_objs.itervalues())
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.tables)
        children.extend(self.artists)
        children.extend(self.images)
        if self.legend_ is not None:
            children.append(self.legend_)
        children.extend(self.collections)
        children.append(self.title)
        children.append(self.patch)
        children.extend(self.spines.itervalues())
        return children

    def contains(self, mouseevent):
        """
        Test whether the mouse event occured in the axes.

        Returns *True* / *False*, {}

        """
        if callable(self._contains):
            return self._contains(self, mouseevent)

        return self.patch.contains(mouseevent)

    def contains_point(self, point):
        """
        Returns *True* if the point (tuple of x,y) is inside the axes
        (the area defined by the its patch). A pixel coordinate is
        required.

        """
        return self.patch.contains_point(point, radius=1.0)

    #### Labelling

    def get_title(self):
        """
        Get the title text string.
        """
        return self.title.get_text()

    @docstring.dedent_interpd
    def set_title(self, label, fontdict=None, **kwargs):
        """
        Call signature::

          set_title(label, fontdict=None, **kwargs):

        Set the title for the axes.

        kwargs are Text properties:
        %(Text)s

        ACCEPTS: str

        .. seealso::

            :meth:`text`
                for information on how override and the optional args work
        """
        default = {
            'fontsize':rcParams['axes.titlesize'],
            'verticalalignment' : 'baseline',
            'horizontalalignment' : 'center'
            }

        self.title.set_text(label)
        self.title.update(default)
        if fontdict is not None:
            self.title.update(fontdict)
        self.title.update(kwargs)
        return self.title

    def _get_legend_handles(self, legend_handler_map=None):
        "return artists that will be used as handles for legend"
        handles_original = self.lines + self.patches + \
                           self.collections + self.containers

        # collections
        handler_map = mlegend.Legend.get_default_handler_map()

        if legend_handler_map is not None:
            handler_map = handler_map.copy()
            handler_map.update(legend_handler_map)

        handles = []
        for h in handles_original:
            if h.get_label() == "_nolegend_": #.startswith('_'):
                continue
            if mlegend.Legend.get_legend_handler(handler_map, h):
                handles.append(h)

        return handles


    def get_legend_handles_labels(self, legend_handler_map=None):
        """
        Return handles and labels for legend

        ``ax.legend()`` is equivalent to ::

          h, l = ax.get_legend_handles_labels()
          ax.legend(h, l)

        """

        handles = []
        labels = []
        for handle in self._get_legend_handles(legend_handler_map):
            label = handle.get_label()
            #if (label is not None and label != '' and not label.startswith('_')):
            if label and not label.startswith('_'):
                handles.append(handle)
                labels.append(label)

        return handles, labels

    def get_default_bbox_extra_artists(self):
        bbox_extra_artists = [t for t in self.texts if t.get_visible()]
        if self.legend_:
            bbox_extra_artists.append(self.legend_)
        return bbox_extra_artists

    def minorticks_on(self):
        'Add autoscaling minor ticks to the axes.'
        for name, ax in self._axis_objs.iteritems():
            if ax.get_scale() == 'log':
                s = ax._scale
                ax.set_minor_locator(mticker.LogLocator(s.base, s.subs))
            else:
                ax.set_minor_locator(mticker.AutoMinorLocator())

    def minorticks_off(self):
        """Remove minor ticks from the axes."""
        for name, ax in self._axis_objs.iteritems():
            ax.set_minor_locator(mticker.NullLocator())


