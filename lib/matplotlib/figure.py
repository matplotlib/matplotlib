"""
The figure module provides the top-level
:class:`~matplotlib.artist.Artist`, the :class:`Figure`, which
contains all the plot elements.  The following classes are defined

:class:`SubplotParams`
    control the default spacing of the subplots

:class:`Figure`
    Top level container for all plot elements.

"""

import logging
from numbers import Integral
import warnings

import numpy as np

from matplotlib import rcParams
from matplotlib import backends, docstring
from matplotlib import __version__ as _mpl_version
from matplotlib import get_backend

import matplotlib.artist as martist
from matplotlib.artist import Artist, allow_rasterization

import matplotlib.cbook as cbook

from matplotlib.cbook import Stack, iterable

from matplotlib import image as mimage
from matplotlib.image import FigureImage

import matplotlib.colorbar as cbar

from matplotlib.axes import Axes, SubplotBase, subplot_class_factory
from matplotlib.blocking_input import BlockingMouseInput, BlockingKeyMouseInput
from matplotlib.gridspec import GridSpec
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.projections import (get_projection_names,
                                    process_projection_requirements)
from matplotlib.text import Text, TextWithDash
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)
import matplotlib._layoutbox as layoutbox
from matplotlib.backend_bases import NonGuiException

_log = logging.getLogger(__name__)

docstring.interpd.update(projection_names=get_projection_names())


def _stale_figure_callback(self, val):
    if self.figure:
        self.figure.stale = val


class AxesStack(Stack):
    """
    Specialization of the `.Stack` to handle all tracking of
    `~matplotlib.axes.Axes` in a `.Figure`.
    This stack stores ``key, (ind, axes)`` pairs, where:

        * **key** should be a hash of the args and kwargs
          used in generating the Axes.
        * **ind** is a serial number for tracking the order
          in which axes were added.

    The AxesStack is a callable, where ``ax_stack()`` returns
    the current axes. Alternatively the :meth:`current_key_axes` will
    return the current key and associated axes.

    """
    def __init__(self):
        Stack.__init__(self)
        self._ind = 0

    def as_list(self):
        """
        Return a list of the Axes instances that have been added to the figure.
        """
        ia_list = [a for k, a in self._elements]
        ia_list.sort()
        return [a for i, a in ia_list]

    def get(self, key):
        """
        Return the Axes instance that was added with *key*.
        If it is not present, return *None*.
        """
        item = dict(self._elements).get(key)
        if item is None:
            return None
        cbook.warn_deprecated(
            "2.1",
            "Adding an axes using the same arguments as a previous axes "
            "currently reuses the earlier instance.  In a future version, "
            "a new instance will always be created and returned.  Meanwhile, "
            "this warning can be suppressed, and the future behavior ensured, "
            "by passing a unique label to each axes instance.")
        return item[1]

    def _entry_from_axes(self, e):
        ind, k = {a: (ind, k) for k, (ind, a) in self._elements}[e]
        return (k, (ind, e))

    def remove(self, a):
        """Remove the axes from the stack."""
        Stack.remove(self, self._entry_from_axes(a))

    def bubble(self, a):
        """
        Move the given axes, which must already exist in the
        stack, to the top.
        """
        return Stack.bubble(self, self._entry_from_axes(a))

    def add(self, key, a):
        """
        Add Axes *a*, with key *key*, to the stack, and return the stack.

        If *key* is unhashable, replace it by a unique, arbitrary object.

        If *a* is already on the stack, don't add it again, but
        return *None*.
        """
        # All the error checking may be unnecessary; but this method
        # is called so seldom that the overhead is negligible.
        if not isinstance(a, Axes):
            raise ValueError("second argument, {!r}, is not an Axes".format(a))
        try:
            hash(key)
        except TypeError:
            key = object()

        a_existing = self.get(key)
        if a_existing is not None:
            Stack.remove(self, (key, a_existing))
            warnings.warn(
                "key {!r} already existed; Axes is being replaced".format(key))
            # I don't think the above should ever happen.

        if a in self:
            return None
        self._ind += 1
        return Stack.push(self, (key, (self._ind, a)))

    def current_key_axes(self):
        """
        Return a tuple of ``(key, axes)`` for the active axes.

        If no axes exists on the stack, then returns ``(None, None)``.
        """
        if not len(self._elements):
            return self._default, self._default
        else:
            key, (index, axes) = self._elements[self._pos]
            return key, axes

    def __call__(self):
        return self.current_key_axes()[1]

    def __contains__(self, a):
        return a in self.as_list()


class SubplotParams(object):
    """
    A class to hold the parameters for a subplot.
    """
    def __init__(self, left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None):
        """
        All dimensions are fractions of the figure width or height.
        Defaults are given by :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float
            The left side of the subplots of the figure.

        right : float
            The right side of the subplots of the figure.

        bottom : float
            The bottom of the subplots of the figure.

        top : float
            The top of the subplots of the figure.

        wspace : float
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.

        hspace : float
            The amount of height reserved for space between subplots,
            expressed as a fraction of the average axis height.
        """
        self.validate = True
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self, left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
        thisleft = getattr(self, 'left', None)
        thisright = getattr(self, 'right', None)
        thistop = getattr(self, 'top', None)
        thisbottom = getattr(self, 'bottom', None)
        thiswspace = getattr(self, 'wspace', None)
        thishspace = getattr(self, 'hspace', None)

        self._update_this('left', left)
        self._update_this('right', right)
        self._update_this('bottom', bottom)
        self._update_this('top', top)
        self._update_this('wspace', wspace)
        self._update_this('hspace', hspace)

        def reset():
            self.left = thisleft
            self.right = thisright
            self.top = thistop
            self.bottom = thisbottom
            self.wspace = thiswspace
            self.hspace = thishspace

        if self.validate:
            if self.left >= self.right:
                reset()
                raise ValueError('left cannot be >= right')

            if self.bottom >= self.top:
                reset()
                raise ValueError('bottom cannot be >= top')

    def _update_this(self, s, val):
        if val is None:
            val = getattr(self, s, None)
            if val is None:
                key = 'figure.subplot.' + s
                val = rcParams[key]

        setattr(self, s, val)


class Figure(Artist):
    """
    The top level container for all the plot elements.

    The Figure instance supports callbacks through a *callbacks* attribute
    which is a `.CallbackRegistry` instance.  The events you can connect to
    are 'dpi_changed', and the callback will be called with ``func(fig)`` where
    fig is the `Figure` instance.

    Attributes
    ----------
    patch
        The `.Rectangle` instance representing the figure patch.

    suppressComposite
        For multiple figure images, the figure will make composite images
        depending on the renderer option_image_nocomposite function.  If
        *suppressComposite* is a boolean, this will override the renderer.
    """

    def __str__(self):
        return "Figure(%gx%g)" % tuple(self.bbox.size)

    def __repr__(self):
        return "<{clsname} size {h:g}x{w:g} with {naxes} Axes>".format(
            clsname=self.__class__.__name__,
            h=self.bbox.size[0], w=self.bbox.size[1],
            naxes=len(self.axes),
        )

    def __init__(self,
                 figsize=None,
                 dpi=None,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 subplotpars=None,  # default to rc
                 tight_layout=None,  # default to rc figure.autolayout
                 constrained_layout=None,  # default to rc
                                          #figure.constrained_layout.use
                 ):
        """
        Parameters
        ----------
        figsize : 2-tuple of floats, default: :rc:`figure.figsize`
            Figure dimension ``(width, height)`` in inches.

        dpi : float, default: :rc:`figure.dpi`
            Dots per inch.

        facecolor : default: :rc:`figure.facecolor`
            The figure patch facecolor.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure frame.

        subplotpars : :class:`SubplotParams`
            Subplot parameters. If not given, the default subplot
            parameters :rc:`figure.subplot.*` are used.

        tight_layout : bool or dict, default: :rc:`figure.autolayout`
            If ``False`` use *subplotpars*. If ``True`` adjust subplot
            parameters using `.tight_layout` with default padding.
            When providing a dict containing the keys ``pad``, ``w_pad``,
            ``h_pad``, and ``rect``, the default `.tight_layout` paddings
            will be overridden.

        constrained_layout : bool
            If ``True`` use constrained layout to adjust positioning of plot
            elements.  Like ``tight_layout``, but designed to be more
            flexible.  See
            :doc:`/tutorials/intermediate/constrainedlayout_guide`
            for examples.  (Note: does not work with :meth:`.subplot` or
            :meth:`.subplot2grid`.)
            Defaults to :rc:`figure.constrained_layout.use`.
        """
        Artist.__init__(self)
        # remove the non-figure artist _axes property
        # as it makes no sense for a figure to be _in_ an axes
        # this is used by the property methods in the artist base class
        # which are over-ridden in this class
        del self._axes
        self.callbacks = cbook.CallbackRegistry()

        if figsize is None:
            figsize = rcParams['figure.figsize']
        if dpi is None:
            dpi = rcParams['figure.dpi']
        if facecolor is None:
            facecolor = rcParams['figure.facecolor']
        if edgecolor is None:
            edgecolor = rcParams['figure.edgecolor']
        if frameon is None:
            frameon = rcParams['figure.frameon']

        if not np.isfinite(figsize).all():
            raise ValueError('figure size must be finite not '
                             '{}'.format(figsize))
        self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)

        self.dpi_scale_trans = Affine2D().scale(dpi, dpi)
        # do not use property as it will trigger
        self._dpi = dpi
        self.bbox = TransformedBbox(self.bbox_inches, self.dpi_scale_trans)

        self.frameon = frameon

        self.transFigure = BboxTransformTo(self.bbox)

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
        self._set_artist_props(self.patch)
        self.patch.set_aa(False)

        self.canvas = None
        self._suptitle = None

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars
        # constrained_layout:
        self._layoutbox = None
        # set in set_constrained_layout_pads()
        self.set_constrained_layout(constrained_layout)

        self.set_tight_layout(tight_layout)

        self._axstack = AxesStack()  # track all figure axes and current axes
        self.clf()
        self._cachedRenderer = None

        # groupers to keep track of x and y labels we want to align.
        # see self.align_xlabels and self.align_ylabels and
        # axis._get_tick_boxes_siblings
        self._align_xlabel_grp = cbook.Grouper()
        self._align_ylabel_grp = cbook.Grouper()

        # list of child gridspecs for this figure
        self._gridspecs = []

    # TODO: I'd like to dynamically add the _repr_html_ method
    # to the figure in the right context, but then IPython doesn't
    # use it, for some reason.

    def _repr_html_(self):
        # We can't use "isinstance" here, because then we'd end up importing
        # webagg unconditiionally.
        if (self.canvas is not None and
                'WebAgg' in self.canvas.__class__.__name__):
            from matplotlib.backends import backend_webagg
            return backend_webagg.ipython_inline_display(self)

    def show(self, warn=True):
        """
        If using a GUI backend with pyplot, display the figure window.

        If the figure was not created using
        :func:`~matplotlib.pyplot.figure`, it will lack a
        :class:`~matplotlib.backend_bases.FigureManagerBase`, and
        will raise an AttributeError.

        Parameters
        ----------
        warn : bool
            If ``True`` and we are not running headless (i.e. on Linux with an
            unset DISPLAY), issue warning when called on a non-GUI backend.
        """
        try:
            manager = getattr(self.canvas, 'manager')
        except AttributeError as err:
            raise AttributeError("%s\n"
                                 "Figure.show works only "
                                 "for figures managed by pyplot, normally "
                                 "created by pyplot.figure()." % err)

        if manager is not None:
            try:
                manager.show()
                return
            except NonGuiException:
                pass
        if (backends._get_running_interactive_framework() != "headless"
                and warn):
            warnings.warn('Matplotlib is currently using %s, which is a '
                          'non-GUI backend, so cannot show the figure.'
                          % get_backend())

    def _get_axes(self):
        return self._axstack.as_list()

    axes = property(fget=_get_axes,
                    doc="List of axes in the Figure. You can access the "
                        "axes in the Figure through this list. "
                        "Do not modify the list itself. Instead, use "
                        "`~Figure.add_axes`, `~.Figure.subplot` or "
                        "`~.Figure.delaxes` to add or remove an axes.")

    def _get_dpi(self):
        return self._dpi

    def _set_dpi(self, dpi, forward=True):
        """
        Parameters
        ----------
        dpi : float

        forward : bool
            Passed on to `~.Figure.set_size_inches`
        """
        self._dpi = dpi
        self.dpi_scale_trans.clear().scale(dpi, dpi)
        w, h = self.get_size_inches()
        self.set_size_inches(w, h, forward=forward)
        self.callbacks.process('dpi_changed', self)

    dpi = property(_get_dpi, _set_dpi, doc="The resolution in dots per inch.")

    def get_tight_layout(self):
        """Return whether `.tight_layout` is called when drawing."""
        return self._tight

    def set_tight_layout(self, tight):
        """
        Set whether and how `.tight_layout` is called when drawing.

        Parameters
        ----------
        tight : bool or dict with keys "pad", "w_pad", "h_pad", "rect" or None
            If a bool, sets whether to call `.tight_layout` upon drawing.
            If ``None``, use the ``figure.autolayout`` rcparam instead.
            If a dict, pass it as kwargs to `.tight_layout`, overriding the
            default paddings.
        """
        if tight is None:
            tight = rcParams['figure.autolayout']
        self._tight = bool(tight)
        self._tight_parameters = tight if isinstance(tight, dict) else {}
        self.stale = True

    def get_constrained_layout(self):
        """
        Return a boolean: True means constrained layout is being used.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.
        """
        return self._constrained

    def set_constrained_layout(self, constrained):
        """
        Set whether ``constrained_layout`` is used upon drawing. If None,
        the rcParams['figure.constrained_layout.use'] value will be used.

        When providing a dict containing the keys `w_pad`, `h_pad`
        the default ``constrained_layout`` paddings will be
        overridden.  These pads are in inches and default to 3.0/72.0.
        ``w_pad`` is the width padding and ``h_pad`` is the height padding.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

        Parameters
        ----------
        constrained : bool or dict or None
        """
        self._constrained_layout_pads = dict()
        self._constrained_layout_pads['w_pad'] = None
        self._constrained_layout_pads['h_pad'] = None
        self._constrained_layout_pads['wspace'] = None
        self._constrained_layout_pads['hspace'] = None
        if constrained is None:
            constrained = rcParams['figure.constrained_layout.use']
        self._constrained = bool(constrained)
        if isinstance(constrained, dict):
            self.set_constrained_layout_pads(**constrained)
        else:
            self.set_constrained_layout_pads()

        self.stale = True

    def set_constrained_layout_pads(self, **kwargs):
        """
        Set padding for ``constrained_layout``.  Note the kwargs can be passed
        as a dictionary ``fig.set_constrained_layout(**paddict)``.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

        Parameters
        ----------

        w_pad : scalar
            Width padding in inches.  This is the pad around axes
            and is meant to make sure there is enough room for fonts to
            look good.  Defaults to 3 pts = 0.04167 inches

        h_pad : scalar
            Height padding in inches. Defaults to 3 pts.

        wspace: scalar
            Width padding between subplots, expressed as a fraction of the
            subplot width.  The total padding ends up being w_pad + wspace.

        hspace: scalar
            Height padding between subplots, expressed as a fraction of the
            subplot width. The total padding ends up being h_pad + hspace.

        """

        todo = ['w_pad', 'h_pad', 'wspace', 'hspace']
        for td in todo:
            if td in kwargs and kwargs[td] is not None:
                self._constrained_layout_pads[td] = kwargs[td]
            else:
                self._constrained_layout_pads[td] = (
                    rcParams['figure.constrained_layout.' + td])

    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of `w_pad, h_pad` in inches and
        `wspace` and `hspace` as fractions of the subplot.

        See :doc:`/tutorials/intermediate/constrainedlayout_guide`.

        Parameters
        ----------

        relative : boolean
            If `True`, then convert from inches to figure relative.
        """
        w_pad = self._constrained_layout_pads['w_pad']
        h_pad = self._constrained_layout_pads['h_pad']
        wspace = self._constrained_layout_pads['wspace']
        hspace = self._constrained_layout_pads['hspace']

        if relative and ((w_pad is not None) or (h_pad is not None)):
            renderer0 = layoutbox.get_renderer(self)
            dpi = renderer0.dpi
            w_pad = w_pad * dpi / renderer0.width
            h_pad = h_pad * dpi / renderer0.height

        return w_pad, h_pad, wspace, hspace

    def autofmt_xdate(self, bottom=0.2, rotation=30, ha='right', which=None):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared xaxes where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        Parameters
        ----------
        bottom : scalar
            The bottom of the subplots for :meth:`subplots_adjust`.

        rotation : angle in degrees
            The rotation of the xtick labels.

        ha : string
            The horizontal alignment of the xticklabels.

        which : {None, 'major', 'minor', 'both'}
            Selects which ticklabels to rotate. Default is None which works
            the same as major.
        """
        allsubplots = all(hasattr(ax, 'is_last_row') for ax in self.axes)
        if len(self.axes) == 1:
            for label in self.axes[0].get_xticklabels(which=which):
                label.set_ha(ha)
                label.set_rotation(rotation)
        else:
            if allsubplots:
                for ax in self.get_axes():
                    if ax.is_last_row():
                        for label in ax.get_xticklabels(which=which):
                            label.set_ha(ha)
                            label.set_rotation(rotation)
                    else:
                        for label in ax.get_xticklabels(which=which):
                            label.set_visible(False)
                        ax.set_xlabel('')

        if allsubplots:
            self.subplots_adjust(bottom=bottom)
        self.stale = True

    def get_children(self):
        """Get a list of artists contained in the figure."""
        children = [self.patch]
        children.extend(self.artists)
        children.extend(self.axes)
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.images)
        children.extend(self.legends)
        return children

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the figure.

        Returns
        -------
            bool, {}
        """
        if callable(self._contains):
            return self._contains(self, mouseevent)
        inside = self.bbox.contains(mouseevent.x, mouseevent.y)
        return inside, {}

    def get_window_extent(self, *args, **kwargs):
        """
        Return the figure bounding box in display space. Arguments are ignored.
        """
        return self.bbox

    def suptitle(self, t, **kwargs):
        """
        Add a centered title to the figure.

        Parameters
        ----------
        t : str
            The title text.

        x : float, default 0.5
            The x location of the text in figure coordinates.

        y : float, default 0.98
            The y location of the text in figure coordinates.

        horizontalalignment, ha : {'center', 'left', right'}, default: 'center'
            The horizontal alignment of the text relative to (*x*, *y*).

        verticalalignment, va : {'top', 'center', 'bottom', 'baseline'}, \
default: 'top'
            The vertical alignment of the text relative to (*x*, *y*).

        fontsize, size : default: :rc:`figure.titlesize`
            The font size of the text. See `.Text.set_size` for possible
            values.

        fontweight, weight : default: :rc:`figure.titleweight`
            The font weight of the text. See `.Text.set_weight` for possible
            values.


        Returns
        -------
            text
                The `.Text` instance of the title.


        Other Parameters
        ----------------
        fontproperties : None or dict, optional
            A dict of font properties. If *fontproperties* is given the
            default values for font size and weight are taken from the
            `FontProperties` defaults. :rc:`figure.titlesize` and
            :rc:`figure.titleweight` are ignored in this case.

        **kwargs
            Additional kwargs are :class:`matplotlib.text.Text` properties.


        Examples
        --------

        >>> fig.suptitle('This is the figure title', fontsize=12)
        """
        manual_position = ('x' in kwargs or 'y' in kwargs)

        x = kwargs.pop('x', 0.5)
        y = kwargs.pop('y', 0.98)

        if 'horizontalalignment' not in kwargs and 'ha' not in kwargs:
            kwargs['horizontalalignment'] = 'center'
        if 'verticalalignment' not in kwargs and 'va' not in kwargs:
            kwargs['verticalalignment'] = 'top'

        if 'fontproperties' not in kwargs:
            if 'fontsize' not in kwargs and 'size' not in kwargs:
                kwargs['size'] = rcParams['figure.titlesize']
            if 'fontweight' not in kwargs and 'weight' not in kwargs:
                kwargs['weight'] = rcParams['figure.titleweight']

        sup = self.text(x, y, t, **kwargs)
        if self._suptitle is not None:
            self._suptitle.set_text(t)
            self._suptitle.set_position((x, y))
            self._suptitle.update_from(sup)
            sup.remove()
        else:
            self._suptitle = sup
            self._suptitle._layoutbox = None
            if self._layoutbox is not None and not manual_position:
                w_pad, h_pad, wspace, hspace =  \
                        self.get_constrained_layout_pads(relative=True)
                figlb = self._layoutbox
                self._suptitle._layoutbox = layoutbox.LayoutBox(
                        parent=figlb, artist=self._suptitle,
                        name=figlb.name+'.suptitle')
                # stack the suptitle on top of all the children.
                # Some day this should be on top of all the children in the
                # gridspec only.
                for child in figlb.children:
                    if child is not self._suptitle._layoutbox:
                        layoutbox.vstack([self._suptitle._layoutbox,
                                          child],
                                         padding=h_pad*2., strength='required')
        self.stale = True
        return self._suptitle

    def set_canvas(self, canvas):
        """
        Set the canvas that contains the figure

        Parameters
        ----------
        canvas : FigureCanvas
        """
        self.canvas = canvas

    def figimage(self, X, xo=0, yo=0, alpha=None, norm=None, cmap=None,
                 vmin=None, vmax=None, origin=None, resize=False, **kwargs):
        """
        Add a non-resampled image to the figure.

        The image is attached to the lower or upper left corner depending on
        *origin*.

        Parameters
        ----------
        X
            The image data. This is an array of one of the following shapes:

            - MxN: luminance (grayscale) values
            - MxNx3: RGB values
            - MxNx4: RGBA values

        xo, yo : int
            The *x*/*y* image offset in pixels.

        alpha : None or float
            The alpha blending value.

        norm : :class:`matplotlib.colors.Normalize`
            A :class:`.Normalize` instance to map the luminance to the
            interval [0, 1].

        cmap : str or :class:`matplotlib.colors.Colormap`
            The colormap to use. Default: :rc:`image.cmap`.

        vmin, vmax : scalar
            If *norm* is not given, these values set the data limits for the
            colormap.

        origin : {'upper', 'lower'}
            Indicates where the [0, 0] index of the array is in the upper left
            or lower left corner of the axes. Defaults to :rc:`image.origin`.

        resize : bool
            If *True*, resize the figure to match the given image size.

        Returns
        -------
        :class:`matplotlib.image.FigureImage`

        Other Parameters
        ----------------
        **kwargs
            Additional kwargs are `.Artist` kwargs passed on to `.FigureImage`.

        Notes
        -----
        figimage complements the axes image
        (:meth:`~matplotlib.axes.Axes.imshow`) which will be resampled
        to fit the current axes.  If you want a resampled image to
        fill the entire figure, you can define an
        :class:`~matplotlib.axes.Axes` with extent [0,0,1,1].


        Examples::

            f = plt.figure()
            nx = int(f.get_figwidth() * f.dpi)
            ny = int(f.get_figheight() * f.dpi)
            data = np.random.random((ny, nx))
            f.figimage(data)
            plt.show()

        """
        if resize:
            dpi = self.get_dpi()
            figsize = [x / dpi for x in (X.shape[1], X.shape[0])]
            self.set_size_inches(figsize, forward=True)

        im = FigureImage(self, cmap, norm, xo, yo, origin, **kwargs)
        im.stale_callback = _stale_figure_callback

        im.set_array(X)
        im.set_alpha(alpha)
        if norm is None:
            im.set_clim(vmin, vmax)
        self.images.append(im)
        im._remove_method = self.images.remove
        self.stale = True
        return im

    def set_size_inches(self, w, h=None, forward=True):
        """Set the figure size in inches.

        Call signatures::

             fig.set_size_inches(w, h)  # OR
             fig.set_size_inches((w, h))

        optional kwarg *forward=True* will cause the canvas size to be
        automatically updated; e.g., you can resize the figure window
        from the shell

        ACCEPTS: a (w, h) tuple with w, h in inches

        See Also
        --------
        matplotlib.Figure.get_size_inches
        """

        # the width and height have been passed in as a tuple to the first
        # argument, so unpack them
        if h is None:
            w, h = w
        if not all(np.isfinite(_) for _ in (w, h)):
            raise ValueError('figure size must be finite not '
                             '({}, {})'.format(w, h))
        self.bbox_inches.p1 = w, h

        if forward:
            canvas = getattr(self, 'canvas')
            if canvas is not None:
                ratio = getattr(self.canvas, '_dpi_ratio', 1)
                dpival = self.dpi / ratio
                canvasw = w * dpival
                canvash = h * dpival
                manager = getattr(self.canvas, 'manager', None)
                if manager is not None:
                    manager.resize(int(canvasw), int(canvash))
        self.stale = True

    def get_size_inches(self):
        """
        Returns the current size of the figure in inches.

        Returns
        -------
        size : ndarray
           The size (width, height) of the figure in inches.

        See Also
        --------
        matplotlib.Figure.set_size_inches
        """
        return np.array(self.bbox_inches.p1)

    def get_edgecolor(self):
        """Get the edge color of the Figure rectangle."""
        return self.patch.get_edgecolor()

    def get_facecolor(self):
        """Get the face color of the Figure rectangle."""
        return self.patch.get_facecolor()

    def get_figwidth(self):
        """Return the figure width as a float."""
        return self.bbox_inches.width

    def get_figheight(self):
        """Return the figure height as a float."""
        return self.bbox_inches.height

    def get_dpi(self):
        """Return the resolution in dots per inch as a float."""
        return self.dpi

    def get_frameon(self):
        """Return whether the figure frame will be drawn."""
        return self.frameon

    def set_edgecolor(self, color):
        """
        Set the edge color of the Figure rectangle.

        Parameters
        ----------
        color : color
        """
        self.patch.set_edgecolor(color)

    def set_facecolor(self, color):
        """
        Set the face color of the Figure rectangle.

        Parameters
        ----------
        color : color
        """
        self.patch.set_facecolor(color)

    def set_dpi(self, val):
        """
        Set the resolution of the figure in dots-per-inch.

        Parameters
        ----------
        val : float
        """
        self.dpi = val
        self.stale = True

    def set_figwidth(self, val, forward=True):
        """
        Set the width of the figure in inches.

        .. ACCEPTS: float
        """
        self.set_size_inches(val, self.get_figheight(), forward=forward)

    def set_figheight(self, val, forward=True):
        """
        Set the height of the figure in inches.

        .. ACCEPTS: float
        """
        self.set_size_inches(self.get_figwidth(), val, forward=forward)

    def set_frameon(self, b):
        """
        Set whether the figure frame (background) is displayed or invisible.

        Parameters
        ----------
        b : bool
        """
        self.frameon = b
        self.stale = True

    def delaxes(self, ax):
        """
        Remove the `~matplotlib.axes.Axes` *ax* from the figure and update the
        current axes.
        """
        self._axstack.remove(ax)
        for func in self._axobservers:
            func(self)
        self.stale = True

    def _make_key(self, *args, **kwargs):
        """Make a hashable key out of args and kwargs."""

        def fixitems(items):
            # items may have arrays and lists in them, so convert them
            # to tuples for the key
            ret = []
            for k, v in items:
                # some objects can define __getitem__ without being
                # iterable and in those cases the conversion to tuples
                # will fail. So instead of using the iterable(v) function
                # we simply try and convert to a tuple, and proceed if not.
                try:
                    v = tuple(v)
                except Exception:
                    pass
                ret.append((k, v))
            return tuple(ret)

        def fixlist(args):
            ret = []
            for a in args:
                if iterable(a):
                    a = tuple(a)
                ret.append(a)
            return tuple(ret)

        key = fixlist(args), fixitems(kwargs.items())
        return key

    def add_artist(self, artist, clip=False):
        """
        Add any :class:`~matplotlib.artist.Artist` to the figure.

        Usually artists are added to axes objects using
        :meth:`matplotlib.axes.Axes.add_artist`, but use this method in the
        rare cases that adding directly to the figure is necessary.

        Parameters
        ----------
        artist : `~matplotlib.artist.Artist`
            The artist to add to the figure. If the added artist has no
            transform previously set, its transform will be set to
            ``figure.transFigure``.
        clip : bool, optional, default ``False``
            An optional parameter ``clip`` determines whether the added artist
            should be clipped by the figure patch. Default is *False*,
            i.e. no clipping.

        Returns
        -------
        artist : The added `~matplotlib.artist.Artist`
        """
        artist.set_figure(self)
        self.artists.append(artist)
        artist._remove_method = self.artists.remove

        if not artist.is_transform_set():
            artist.set_transform(self.transFigure)

        if clip:
            artist.set_clip_path(self.patch)

        self.stale = True
        return artist

    @docstring.dedent_interpd
    def add_axes(self, *args, **kwargs):
        """
        Add an axes to the figure.

        Call signatures::

            add_axes(rect, projection=None, polar=False, **kwargs)
            add_axes(ax)

        Parameters
        ----------

        rect : sequence of float
            The dimensions [left, bottom, width, height] of the new axes. All
            quantities are in fractions of figure width and height.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.

        polar : boolean, optional
            If True, equivalent to projection='polar'.

        sharex, sharey : `~.axes.Axes`, optional
            Share the x or y `~matplotlib.axis` with sharex and/or sharey.
            The axis will have the same limits, ticks, and scale as the axis
            of the shared axes.

        label : str
            A label for the returned axes.

        Other Parameters
        ----------------
        **kwargs
            This method also takes the keyword arguments for
            the returned axes class. The keyword arguments for the
            rectilinear axes class `~.axes.Axes` can be found in
            the following table but there might also be other keyword
            arguments if another projection is used, see the actual axes
            class.
            %(Axes)s

        Returns
        -------
        axes : `~.axes.Axes` (or a subclass of `~.axes.Axes`)
            The returned axes class depends on the projection used. It is
            `~.axes.Axes` if rectilinear projection are used and
            `.projections.polar.PolarAxes` if polar projection
            are used.

        Notes
        -----
        If the figure already has an axes with key (*args*,
        *kwargs*) then it will simply make that axes current and
        return it.  This behavior is deprecated. Meanwhile, if you do
        not want this behavior (i.e., you want to force the creation of a
        new axes), you must use a unique set of args and kwargs.  The axes
        *label* attribute has been exposed for this purpose: if you want
        two axes that are otherwise identical to be added to the figure,
        make sure you give them unique labels.

        In rare circumstances, `.add_axes` may be called with a single
        argument, a axes instance already created in the present figure but
        not in the figure's list of axes.

        See Also
        --------
        .Figure.add_subplot
        .pyplot.subplot
        .pyplot.axes
        .Figure.subplots
        .pyplot.subplots

        Examples
        --------
        Some simple examples::

            rect = l, b, w, h
            fig = plt.figure(1)
            fig.add_axes(rect,label=label1)
            fig.add_axes(rect,label=label2)
            fig.add_axes(rect, frameon=False, facecolor='g')
            fig.add_axes(rect, polar=True)
            ax=fig.add_axes(rect, projection='polar')
            fig.delaxes(ax)
            fig.add_axes(ax)
        """

        if not len(args):
            return

        # shortcut the projection "key" modifications later on, if an axes
        # with the exact args/kwargs exists, return it immediately.
        key = self._make_key(*args, **kwargs)
        ax = self._axstack.get(key)
        if ax is not None:
            self.sca(ax)
            return ax

        if isinstance(args[0], Axes):
            a = args[0]
            if a.get_figure() is not self:
                raise ValueError(
                    "The Axes must have been created in the present figure")
        else:
            rect = args[0]
            if not np.isfinite(rect).all():
                raise ValueError('all entries in rect must be finite '
                                 'not {}'.format(rect))
            projection_class, kwargs, key = process_projection_requirements(
                self, *args, **kwargs)

            # check that an axes of this type doesn't already exist, if it
            # does, set it as active and return it
            ax = self._axstack.get(key)
            if isinstance(ax, projection_class):
                self.sca(ax)
                return ax

            # create the new axes using the axes class given
            a = projection_class(self, rect, **kwargs)

        self._axstack.add(key, a)
        self.sca(a)
        a._remove_method = self._remove_ax
        self.stale = True
        a.stale_callback = _stale_figure_callback
        return a

    @docstring.dedent_interpd
    def add_subplot(self, *args, **kwargs):
        """
        Add an `~.axes.Axes` to the figure as part of a subplot arrangement.

        Call signatures::

           add_subplot(nrows, ncols, index, **kwargs)
           add_subplot(pos, **kwargs)
           add_subplot(ax)

        Parameters
        ----------
        *args
            Either a 3-digit integer or three separate integers
            describing the position of the subplot. If the three
            integers are *nrows*, *ncols*, and *index* in order, the
            subplot will take the *index* position on a grid with *nrows*
            rows and *ncols* columns. *index* starts at 1 in the upper left
            corner and increases to the right.

            *pos* is a three digit integer, where the first digit is the
            number of rows, the second the number of columns, and the third
            the index of the subplot. i.e. fig.add_subplot(235) is the same as
            fig.add_subplot(2, 3, 5). Note that all integers must be less than
            10 for this form to work.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, optional
            The projection type of the subplot (`~.axes.Axes`). *str* is the
            name of a custom projection, see `~matplotlib.projections`. The
            default None results in a 'rectilinear' projection.

        polar : boolean, optional
            If True, equivalent to projection='polar'.

        sharex, sharey : `~.axes.Axes`, optional
            Share the x or y `~matplotlib.axis` with sharex and/or sharey.
            The axis will have the same limits, ticks, and scale as the axis
            of the shared axes.

        label : str
            A label for the returned axes.

        Other Parameters
        ----------------
        **kwargs
            This method also takes the keyword arguments for
            the returned axes base class. The keyword arguments for the
            rectilinear base class `~.axes.Axes` can be found in
            the following table but there might also be other keyword
            arguments if another projection is used.
            %(Axes)s

        Returns
        -------
        axes : an `.axes.SubplotBase` subclass of `~.axes.Axes` (or a \
               subclass of `~.axes.Axes`)

            The axes of the subplot. The returned axes base class depends on
            the projection used. It is `~.axes.Axes` if rectilinear projection
            are used and `.projections.polar.PolarAxes` if polar projection
            are used. The returned axes is then a subplot subclass of the
            base class.

        Notes
        -----
        If the figure already has a subplot with key (*args*,
        *kwargs*) then it will simply make that subplot current and
        return it.  This behavior is deprecated. Meanwhile, if you do
        not want this behavior (i.e., you want to force the creation of a
        new suplot), you must use a unique set of args and kwargs.  The axes
        *label* attribute has been exposed for this purpose: if you want
        two subplots that are otherwise identical to be added to the figure,
        make sure you give them unique labels.

        In rare circumstances, `.add_subplot` may be called with a single
        argument, a subplot axes instance already created in the
        present figure but not in the figure's list of axes.

        See Also
        --------
        .Figure.add_axes
        .pyplot.subplot
        .pyplot.axes
        .Figure.subplots
        .pyplot.subplots

        Examples
        --------
        ::

            fig=plt.figure(1)
            fig.add_subplot(221)

            # equivalent but more general
            ax1=fig.add_subplot(2, 2, 1)

            # add a subplot with no frame
            ax2=fig.add_subplot(222, frameon=False)

            # add a polar subplot
            fig.add_subplot(223, projection='polar')

            # add a red subplot that share the x-axis with ax1
            fig.add_subplot(224, sharex=ax1, facecolor='red')

            #delete x2 from the figure
            fig.delaxes(ax2)

            #add x2 to the figure again
            fig.add_subplot(ax2)
        """
        if not len(args):
            return

        if len(args) == 1 and isinstance(args[0], Integral):
            if not 100 <= args[0] <= 999:
                raise ValueError("Integer subplot specification must be a "
                                 "three-digit number, not {}".format(args[0]))
            args = tuple(map(int, str(args[0])))

        if isinstance(args[0], SubplotBase):

            a = args[0]
            if a.get_figure() is not self:
                raise ValueError(
                    "The Subplot must have been created in the present figure")
            # make a key for the subplot (which includes the axes object id
            # in the hash)
            key = self._make_key(*args, **kwargs)
        else:
            projection_class, kwargs, key = process_projection_requirements(
                self, *args, **kwargs)

            # try to find the axes with this key in the stack
            ax = self._axstack.get(key)

            if ax is not None:
                if isinstance(ax, projection_class):
                    # the axes already existed, so set it as active & return
                    self.sca(ax)
                    return ax
                else:
                    # Undocumented convenience behavior:
                    # subplot(111); subplot(111, projection='polar')
                    # will replace the first with the second.
                    # Without this, add_subplot would be simpler and
                    # more similar to add_axes.
                    self._axstack.remove(ax)

            a = subplot_class_factory(projection_class)(self, *args, **kwargs)
        self._axstack.add(key, a)
        self.sca(a)
        a._remove_method = self._remove_ax
        self.stale = True
        a.stale_callback = _stale_figure_callback
        return a

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False,
                 squeeze=True, subplot_kw=None, gridspec_kw=None):
        """
        Add a set of subplots to this figure.

        This utility wrapper makes it convenient to create common layouts of
        subplots in a single call.

        Parameters
        ----------
        nrows, ncols : int, optional, default: 1
            Number of rows/columns of the subplot grid.

        sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
            Controls sharing of properties among x (`sharex`) or y (`sharey`)
            axes:

                - True or 'all': x- or y-axis will be shared among all
                  subplots.
                - False or 'none': each subplot x- or y-axis will be
                  independent.
                - 'row': each subplot row will share an x- or y-axis.
                - 'col': each subplot column will share an x- or y-axis.

            When subplots have a shared x-axis along a column, only the x tick
            labels of the bottom subplot are created. Similarly, when subplots
            have a shared y-axis along a row, only the y tick labels of the
            first column subplot are created. To later turn other subplots'
            ticklabels on, use `~matplotlib.axes.Axes.tick_params`.

        squeeze : bool, optional, default: True
            - If True, extra dimensions are squeezed out from the returned
              array of Axes:

                - if only one subplot is constructed (nrows=ncols=1), the
                  resulting single Axes object is returned as a scalar.
                - for Nx1 or 1xM subplots, the returned object is a 1D numpy
                  object array of Axes objects.
                - for NxM, subplots with N>1 and M>1 are returned
                  as a 2D array.

            - If False, no squeezing at all is done: the returned Axes object
              is always a 2D array containing Axes instances, even if it ends
              up being 1x1.

        subplot_kw : dict, optional
            Dict with keywords passed to the
            :meth:`~matplotlib.figure.Figure.add_subplot` call used to create
            each subplot.

        gridspec_kw : dict, optional
            Dict with keywords passed to the
            `~matplotlib.gridspec.GridSpec` constructor used to create
            the grid the subplots are placed on.

        Returns
        -------
        ax : `~.axes.Axes` object or array of Axes objects.
            *ax* can be either a single `~matplotlib.axes.Axes` object or
            an array of Axes objects if more than one subplot was created. The
            dimensions of the resulting array can be controlled with the
            squeeze keyword, see above.

        Examples
        --------
        ::

            # First create some toy data:
            x = np.linspace(0, 2*np.pi, 400)
            y = np.sin(x**2)

            # Create a figure
            plt.figure(1, clear=True)

            # Creates a subplot
            ax = fig.subplots()
            ax.plot(x, y)
            ax.set_title('Simple plot')

            # Creates two subplots and unpacks the output array immediately
            ax1, ax2 = fig.subplots(1, 2, sharey=True)
            ax1.plot(x, y)
            ax1.set_title('Sharing Y axis')
            ax2.scatter(x, y)

            # Creates four polar axes, and accesses them through the
            # returned array
            axes = fig.subplots(2, 2, subplot_kw=dict(polar=True))
            axes[0, 0].plot(x, y)
            axes[1, 1].scatter(x, y)

            # Share a X axis with each column of subplots
            fig.subplots(2, 2, sharex='col')

            # Share a Y axis with each row of subplots
            fig.subplots(2, 2, sharey='row')

            # Share both X and Y axes with all subplots
            fig.subplots(2, 2, sharex='all', sharey='all')

            # Note that this is the same as
            fig.subplots(2, 2, sharex=True, sharey=True)

            See Also
            --------
            .pyplot.subplots
            .Figure.add_subplot
            .pyplot.subplot
            """

        if isinstance(sharex, bool):
            sharex = "all" if sharex else "none"
        if isinstance(sharey, bool):
            sharey = "all" if sharey else "none"
        share_values = ["all", "row", "col", "none"]
        if sharex not in share_values:
            # This check was added because it is very easy to type
            # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
            # In most cases, no error will ever occur, but mysterious behavior
            # will result because what was intended to be the subplot index is
            # instead treated as a bool for sharex.
            if isinstance(sharex, Integral):
                warnings.warn(
                    "sharex argument to subplots() was an integer. "
                    "Did you intend to use subplot() (without 's')?")

            raise ValueError("sharex [%s] must be one of %s" %
                             (sharex, share_values))
        if sharey not in share_values:
            raise ValueError("sharey [%s] must be one of %s" %
                             (sharey, share_values))
        if subplot_kw is None:
            subplot_kw = {}
        if gridspec_kw is None:
            gridspec_kw = {}
        # don't mutate kwargs passed by user...
        subplot_kw = subplot_kw.copy()
        gridspec_kw = gridspec_kw.copy()

        if self.get_constrained_layout():
            gs = GridSpec(nrows, ncols, figure=self, **gridspec_kw)
        else:
            # this should turn constrained_layout off if we don't want it
            gs = GridSpec(nrows, ncols, figure=None, **gridspec_kw)
        self._gridspecs.append(gs)

        # Create array to hold all axes.
        axarr = np.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                shared_with = {"none": None, "all": axarr[0, 0],
                               "row": axarr[row, 0], "col": axarr[0, col]}
                subplot_kw["sharex"] = shared_with[sharex]
                subplot_kw["sharey"] = shared_with[sharey]
                axarr[row, col] = self.add_subplot(gs[row, col], **subplot_kw)

        # turn off redundant tick labeling
        if sharex in ["col", "all"]:
            # turn off all but the bottom row
            for ax in axarr[:-1, :].flat:
                ax.xaxis.set_tick_params(which='both',
                                         labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
        if sharey in ["row", "all"]:
            # turn off all but the first column
            for ax in axarr[:, 1:].flat:
                ax.yaxis.set_tick_params(which='both',
                                         labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)

        if squeeze:
            # Discarding unneeded dimensions that equal 1.  If we only have one
            # subplot, just return it instead of a 1-element array.
            return axarr.item() if axarr.size == 1 else axarr.squeeze()
        else:
            # Returned axis array will be always 2-d, even if nrows=ncols=1.
            return axarr

    def _remove_ax(self, ax):
        def _reset_loc_form(axis):
            axis.set_major_formatter(axis.get_major_formatter())
            axis.set_major_locator(axis.get_major_locator())
            axis.set_minor_formatter(axis.get_minor_formatter())
            axis.set_minor_locator(axis.get_minor_locator())

        def _break_share_link(ax, grouper):
            siblings = grouper.get_siblings(ax)
            if len(siblings) > 1:
                grouper.remove(ax)
                for last_ax in siblings:
                    if ax is not last_ax:
                        return last_ax
            return None

        self.delaxes(ax)
        last_ax = _break_share_link(ax, ax._shared_y_axes)
        if last_ax is not None:
            _reset_loc_form(last_ax.yaxis)

        last_ax = _break_share_link(ax, ax._shared_x_axes)
        if last_ax is not None:
            _reset_loc_form(last_ax.xaxis)

    def clf(self, keep_observers=False):
        """
        Clear the figure.

        Set *keep_observers* to True if, for example,
        a gui widget is tracking the axes in the figure.
        """
        self.suppressComposite = None
        self.callbacks = cbook.CallbackRegistry()

        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.cla()
            self.delaxes(ax)         # removes ax from self._axstack

        toolbar = getattr(self.canvas, 'toolbar', None)
        if toolbar is not None:
            toolbar.update()
        self._axstack.clear()
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts = []
        self.images = []
        self.legends = []
        if not keep_observers:
            self._axobservers = []
        self._suptitle = None
        if self.get_constrained_layout():
            layoutbox.nonetree(self._layoutbox)
        self.stale = True

    def clear(self, keep_observers=False):
        """
        Clear the figure -- synonym for :meth:`clf`.
        """
        self.clf(keep_observers=keep_observers)

    @allow_rasterization
    def draw(self, renderer):
        """
        Render the figure using :class:`matplotlib.backend_bases.RendererBase`
        instance *renderer*.
        """

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = sorted(
            (artist for artist in (self.patches + self.lines + self.artists
                                   + self.images + self.axes + self.texts
                                   + self.legends)
             if not artist.get_animated()),
            key=lambda artist: artist.get_zorder())

        try:
            renderer.open_group('figure')
            if self.get_constrained_layout() and self.axes:
                self.execute_constrained_layout(renderer)
            if self.get_tight_layout() and self.axes:
                try:
                    self.tight_layout(renderer,
                                      **self._tight_parameters)
                except ValueError:
                    pass
                    # ValueError can occur when resizing a window.

            if self.frameon:
                self.patch.draw(renderer)

            mimage._draw_list_compositing_images(
                renderer, self, artists, self.suppressComposite)

            renderer.close_group('figure')
        finally:
            self.stale = False

        self._cachedRenderer = renderer
        self.canvas.draw_event(renderer)

    def draw_artist(self, a):
        """
        Draw :class:`matplotlib.artist.Artist` instance *a* only.
        This is available only after the figure is drawn.
        """
        if self._cachedRenderer is None:
            raise AttributeError("draw_artist can only be used after an "
                                 "initial draw which caches the renderer")
        a.draw(self._cachedRenderer)

    def get_axes(self):
        """
        Return a list of axes in the Figure. You can access and modify the
        axes in the Figure through this list.

        Do not modify the list itself. Instead, use `~Figure.add_axes`,
        `~.Figure.subplot` or `~.Figure.delaxes` to add or remove an axes.

        Note: This is equivalent to the property `~.Figure.axes`.
        """
        return self.axes

    @docstring.dedent_interpd
    def legend(self, *args, **kwargs):
        """
        Place a legend on the figure.

        To make a legend from existing artists on every axes::

          legend()

        To make a legend for a list of lines and labels::

          legend( (line1, line2, line3),
                  ('label1', 'label2', 'label3'),
                  loc='upper right')

        These can also be specified by keyword::

          legend(handles=(line1, line2, line3),
                labels=('label1', 'label2', 'label3'),
                loc='upper right')

        Parameters
        ----------

        handles : sequence of `.Artist`, optional
            A list of Artists (lines, patches) to be added to the legend.
            Use this together with *labels*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

            The length of handles and labels should be the same in this
            case. If they are not, they are truncated to the smaller length.

        labels : sequence of strings, optional
            A list of labels to show next to the artists.
            Use this together with *handles*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

        Other Parameters
        ----------------

        %(_legend_kw_doc)s

        Returns
        -------
        :class:`matplotlib.legend.Legend` instance

        Notes
        -----
        Not all kinds of artist are supported by the legend command. See
        :doc:`/tutorials/intermediate/legend_guide` for details.
        """

        handles, labels, extra_args, kwargs = mlegend._parse_legend_args(
                self.axes,
                *args,
                **kwargs)
        # check for third arg
        if len(extra_args):
            # cbook.warn_deprecated(
            #     "2.1",
            #     "Figure.legend will accept no more than two "
            #     "positional arguments in the future.  Use "
            #     "'fig.legend(handles, labels, loc=location)' "
            #     "instead.")
            # kwargs['loc'] = extra_args[0]
            # extra_args = extra_args[1:]
            pass
        l = mlegend.Legend(self, handles, labels, *extra_args, **kwargs)
        self.legends.append(l)
        l._remove_method = self.legends.remove
        self.stale = True
        return l

    @docstring.dedent_interpd
    def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
        """
        Add text to figure.

        Parameters
        ----------
        x, y : float
            The position to place the text. By default, this is in figure
            coordinates, floats in [0, 1]. The coordinate system can be changed
            using the *transform* keyword.

        s : str
            The text string.

        fontdict : dictionary, optional, default: None
            A dictionary to override the default text properties. If fontdict
            is None, the defaults are determined by your rc parameters. A
            property in *kwargs* override the same property in fontdict.

        withdash : boolean, optional, default: False
            Creates a `~matplotlib.text.TextWithDash` instance instead of a
            `~matplotlib.text.Text` instance.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            Other miscellaneous text parameters.
            %(Text)s

        Returns
        -------
        text : `~.text.Text`

        See Also
        --------
        .Axes.text
        .pyplot.text
        """
        default = dict(transform=self.transFigure)

        if withdash:
            text = TextWithDash(x=x, y=y, text=s)
        else:
            text = Text(x=x, y=y, text=s)

        text.update(default)
        if fontdict is not None:
            text.update(fontdict)
        text.update(kwargs)

        text.set_figure(self)
        text.stale_callback = _stale_figure_callback

        self.texts.append(text)
        text._remove_method = self.texts.remove
        self.stale = True
        return text

    def _set_artist_props(self, a):
        if a != self:
            a.set_figure(self)
        a.stale_callback = _stale_figure_callback
        a.set_transform(self.transFigure)

    @docstring.dedent_interpd
    def gca(self, **kwargs):
        """
        Get the current axes, creating one if necessary.

        The following kwargs are supported for ensuring the returned axes
        adheres to the given projection etc., and for axes creation if
        the active axes does not exist:

        %(Axes)s

        """
        ckey, cax = self._axstack.current_key_axes()
        # if there exists an axes on the stack see if it maches
        # the desired axes configuration
        if cax is not None:

            # if no kwargs are given just return the current axes
            # this is a convenience for gca() on axes such as polar etc.
            if not kwargs:
                return cax

            # if the user has specified particular projection detail
            # then build up a key which can represent this
            else:
                projection_class, _, key = process_projection_requirements(
                    self, **kwargs)

                # let the returned axes have any gridspec by removing it from
                # the key
                ckey = ckey[1:]
                key = key[1:]

                # if the cax matches this key then return the axes, otherwise
                # continue and a new axes will be created
                if key == ckey and isinstance(cax, projection_class):
                    return cax
                else:
                    warnings.warn('Requested projection is different from '
                                  'current axis projection, creating new axis '
                                  'with requested projection.', stacklevel=2)

        # no axes found, so create one which spans the figure
        return self.add_subplot(1, 1, 1, **kwargs)

    def sca(self, a):
        """Set the current axes to be a and return a."""
        self._axstack.bubble(a)
        for func in self._axobservers:
            func(self)
        return a

    def _gci(self):
        """
        Helper for :func:`~matplotlib.pyplot.gci`. Do not use elsewhere.
        """
        # Look first for an image in the current Axes:
        cax = self._axstack.current_key_axes()[1]
        if cax is None:
            return None
        im = cax._gci()
        if im is not None:
            return im

        # If there is no image in the current Axes, search for
        # one in a previously created Axes.  Whether this makes
        # sense is debatable, but it is the documented behavior.
        for ax in reversed(self.axes):
            im = ax._gci()
            if im is not None:
                return im
        return None

    def __getstate__(self):
        state = super().__getstate__()

        # the axobservers cannot currently be pickled.
        # Additionally, the canvas cannot currently be pickled, but this has
        # the benefit of meaning that a figure can be detached from one canvas,
        # and re-attached to another.
        for attr_to_pop in ('_axobservers', 'show',
                            'canvas', '_cachedRenderer'):
            state.pop(attr_to_pop, None)

        # add version information to the state
        state['__mpl_version__'] = _mpl_version

        # check whether the figure manager (if any) is registered with pyplot
        from matplotlib import _pylab_helpers
        if getattr(self.canvas, 'manager', None) \
                in _pylab_helpers.Gcf.figs.values():
            state['_restore_to_pylab'] = True

        # set all the layoutbox information to None.  kiwisolver objects can't
        # be pickled, so we lose the layout options at this point.
        state.pop('_layoutbox', None)
        # suptitle:
        if self._suptitle is not None:
            self._suptitle._layoutbox = None

        return state

    def __setstate__(self, state):
        version = state.pop('__mpl_version__')
        restore_to_pylab = state.pop('_restore_to_pylab', False)

        if version != _mpl_version:
            import warnings
            warnings.warn("This figure was saved with matplotlib version %s "
                          "and is unlikely to function correctly." %
                          (version, ))

        self.__dict__ = state

        # re-initialise some of the unstored state information
        self._axobservers = []
        self.canvas = None
        self._layoutbox = None

        if restore_to_pylab:
            # lazy import to avoid circularity
            import matplotlib.pyplot as plt
            import matplotlib._pylab_helpers as pylab_helpers
            allnums = plt.get_fignums()
            num = max(allnums) + 1 if allnums else 1
            mgr = plt._backend_mod.new_figure_manager_given_figure(num, self)

            # XXX The following is a copy and paste from pyplot. Consider
            # factoring to pylab_helpers

            if self.get_label():
                mgr.set_window_title(self.get_label())

            # make this figure current on button press event
            def make_active(event):
                pylab_helpers.Gcf.set_active(mgr)

            mgr._cidgcf = mgr.canvas.mpl_connect('button_press_event',
                                                 make_active)

            pylab_helpers.Gcf.set_active(mgr)
            self.number = num

            plt.draw_if_interactive()
        self.stale = True

    def add_axobserver(self, func):
        """Whenever the axes state change, ``func(self)`` will be called."""
        self._axobservers.append(func)

    def savefig(self, fname, *, frameon=None, transparent=None, **kwargs):
        """
        Save the current figure.

        Call signature::

          savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format=None,
                  transparent=False, bbox_inches=None, pad_inches=0.1,
                  frameon=None, metadata=None)

        The output formats available depend on the backend being used.

        Parameters
        ----------

        fname : str or file-like object
            A string containing a path to a filename, or a Python
            file-like object, or possibly some backend-dependent object
            such as :class:`~matplotlib.backends.backend_pdf.PdfPages`.

            If *format* is *None* and *fname* is a string, the output
            format is deduced from the extension of the filename. If
            the filename has no extension, :rc:`savefig.format` is used.

            If *fname* is not a string, remember to specify *format* to
            ensure that the correct backend is used.

        Other Parameters
        ----------------

        dpi : [ *None* | scalar > 0 | 'figure' ]
            The resolution in dots per inch.  If *None*, defaults to
            :rc:`savefig.dpi`.  If 'figure', uses the figure's dpi value.

        quality : [ *None* | 1 <= scalar <= 100 ]
            The image quality, on a scale from 1 (worst) to 95 (best).
            Applicable only if *format* is jpg or jpeg, ignored otherwise.
            If *None*, defaults to :rc:`savefig.jpeg_quality` (95 by default).
            Values above 95 should be avoided; 100 completely disables the
            JPEG quantization stage.

        facecolor : color spec or None, optional
            The facecolor of the figure; if *None*, defaults to
            :rc:`savefig.facecolor`.

        edgecolor : color spec or None, optional
            The edgecolor of the figure; if *None*, defaults to
            :rc:`savefig.edgecolor`

        orientation : {'landscape', 'portrait'}
            Currently only supported by the postscript backend.

        papertype : str
            One of 'letter', 'legal', 'executive', 'ledger', 'a0' through
            'a10', 'b0' through 'b10'. Only supported for postscript
            output.

        format : str
            One of the file extensions supported by the active
            backend.  Most backends support png, pdf, ps, eps and svg.

        transparent : bool
            If *True*, the axes patches will all be transparent; the
            figure patch will also be transparent unless facecolor
            and/or edgecolor are specified via kwargs.
            This is useful, for example, for displaying
            a plot on top of a colored background on a web page.  The
            transparency of these patches will be restored to their
            original values upon exit of this function.

        frameon : bool
            If *True*, the figure patch will be colored, if *False*, the
            figure background will be transparent.  If not provided, the
            rcParam 'savefig.frameon' will be used.

        bbox_inches : str or `~matplotlib.transforms.Bbox`, optional
            Bbox in inches. Only the given portion of the figure is
            saved. If 'tight', try to figure out the tight bbox of
            the figure. If None, use savefig.bbox

        pad_inches : scalar, optional
            Amount of padding around the figure when bbox_inches is
            'tight'. If None, use savefig.pad_inches

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        metadata : dict, optional
            Key/value pairs to store in the image metadata. The supported keys
            and defaults depend on the image format and backend:

            - 'png' with Agg backend: See the parameter ``metadata`` of
              `~.FigureCanvasAgg.print_png`.
            - 'pdf' with pdf backend: See the parameter ``metadata`` of
              `~.backend_pdf.PdfPages`.
            - 'eps' and 'ps' with PS backend: Only 'Creator' is supported.

        """
        kwargs.setdefault('dpi', rcParams['savefig.dpi'])
        if frameon is None:
            frameon = rcParams['savefig.frameon']
        if transparent is None:
            transparent = rcParams['savefig.transparent']

        if transparent:
            kwargs.setdefault('facecolor', 'none')
            kwargs.setdefault('edgecolor', 'none')
            original_axes_colors = []
            for ax in self.axes:
                patch = ax.patch
                original_axes_colors.append((patch.get_facecolor(),
                                             patch.get_edgecolor()))
                patch.set_facecolor('none')
                patch.set_edgecolor('none')
        else:
            kwargs.setdefault('facecolor', rcParams['savefig.facecolor'])
            kwargs.setdefault('edgecolor', rcParams['savefig.edgecolor'])

        if frameon:
            original_frameon = self.get_frameon()
            self.set_frameon(frameon)

        self.canvas.print_figure(fname, **kwargs)

        if frameon:
            self.set_frameon(original_frameon)

        if transparent:
            for ax, cc in zip(self.axes, original_axes_colors):
                ax.patch.set_facecolor(cc[0])
                ax.patch.set_edgecolor(cc[1])

    @docstring.dedent_interpd
    def colorbar(self, mappable, cax=None, ax=None, use_gridspec=True, **kw):
        """
        Create a colorbar for a ScalarMappable instance, *mappable*.

        Documentation for the pyplot thin wrapper:
        %(colorbar_doc)s
        """
        if ax is None:
            ax = self.gca()

        # Store the value of gca so that we can set it back later on.
        current_ax = self.gca()

        if cax is None:
            if use_gridspec and isinstance(ax, SubplotBase)  \
                     and (not self.get_constrained_layout()):
                cax, kw = cbar.make_axes_gridspec(ax, **kw)
            else:
                cax, kw = cbar.make_axes(ax, **kw)

        # need to remove kws that cannot be passed to Colorbar
        NON_COLORBAR_KEYS = ['fraction', 'pad', 'shrink', 'aspect', 'anchor',
                             'panchor']
        cb_kw = {k: v for k, v in kw.items() if k not in NON_COLORBAR_KEYS}
        cb = cbar.colorbar_factory(cax, mappable, **cb_kw)

        self.sca(current_ax)
        self.stale = True
        return cb

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None):
        """
        Update the :class:`SubplotParams` with *kwargs* (defaulting to rc when
        *None*) and update the subplot locations.

        """
        if self.get_constrained_layout():
            self.set_constrained_layout(False)
            warnings.warn("This figure was using constrained_layout==True, "
                          "but that is incompatible with subplots_adjust and "
                          "or tight_layout: setting "
                          "constrained_layout==False. ")
        self.subplotpars.update(left, bottom, right, top, wspace, hspace)
        for ax in self.axes:
            if not isinstance(ax, SubplotBase):
                # Check if sharing a subplots axis
                if isinstance(ax._sharex, SubplotBase):
                    ax._sharex.update_params()
                    ax.set_position(ax._sharex.figbox)
                elif isinstance(ax._sharey, SubplotBase):
                    ax._sharey.update_params()
                    ax.set_position(ax._sharey.figbox)
            else:
                ax.update_params()
                ax.set_position(ax.figbox)
        self.stale = True

    def ginput(self, n=1, timeout=30, show_clicks=True, mouse_add=1,
               mouse_pop=3, mouse_stop=2):
        """
        Blocking call to interact with a figure.

        Wait until the user clicks *n* times on the figure, and return the
        coordinates of each click in a list.

        The buttons used for the various actions (adding points, removing
        points, terminating the inputs) can be overridden via the
        arguments *mouse_add*, *mouse_pop* and *mouse_stop*, that give
        the associated mouse button: 1 for left, 2 for middle, 3 for
        right.

        Parameters
        ----------
        n : int, optional, default: 1
            Number of mouse clicks to accumulate. If negative, accumulate
            clicks until the input is terminated manually.
        timeout : scalar, optional, default: 30
            Number of seconds to wait before timing out. If zero or negative
            will never timeout.
        show_clicks : bool, optional, default: False
            If True, show a red cross at the location of each click.
        mouse_add : int, one of (1, 2, 3), optional, default: 1 (left click)
            Mouse button used to add points.
        mouse_pop : int, one of (1, 2, 3), optional, default: 3 (right click)
            Mouse button used to remove the most recently added point.
        mouse_stop : int, one of (1, 2, 3), optional, default: 2 (middle click)
            Mouse button used to stop input.

        Returns
        -------
        points : list of tuples
            A list of the clicked (x, y) coordinates.

        Notes
        -----
        The keyboard can also be used to select points in case your mouse
        does not have one or more of the buttons.  The delete and backspace
        keys act like right clicking (i.e., remove last point), the enter key
        terminates input and any other key (not already used by the window
        manager) selects a point.
        """

        blocking_mouse_input = BlockingMouseInput(self,
                                                  mouse_add=mouse_add,
                                                  mouse_pop=mouse_pop,
                                                  mouse_stop=mouse_stop)
        return blocking_mouse_input(n=n, timeout=timeout,
                                    show_clicks=show_clicks)

    def waitforbuttonpress(self, timeout=-1):
        """
        Blocking call to interact with the figure.

        This will return True is a key was pressed, False if a mouse
        button was pressed and None if *timeout* was reached without
        either being pressed.

        If *timeout* is negative, does not timeout.
        """

        blocking_input = BlockingKeyMouseInput(self)
        return blocking_input(timeout=timeout)

    def get_default_bbox_extra_artists(self):
        bbox_artists = [artist for artist in self.get_children()
                        if (artist.get_visible() and artist.get_in_layout())]
        for ax in self.axes:
            if ax.get_visible():
                bbox_artists.extend(ax.get_default_bbox_extra_artists())
        # we don't want the figure's patch to influence the bbox calculation
        bbox_artists.remove(self.patch)
        return bbox_artists

    def get_tightbbox(self, renderer, bbox_extra_artists=None):
        """
        Return a (tight) bounding box of the figure in inches.

        Artists that have ``artist.set_in_layout(False)`` are not included
        in the bbox.

        Parameters
        ----------
        renderer : `.RendererBase` instance
            renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        bbox_extra_artists : list of `.Artist` or ``None``
            List of artists to include in the tight bounding box.  If
            ``None`` (default), then all artist children of each axes are
            included in the tight bounding box.

        Returns
        -------
        bbox : `.BboxBase`
            containing the bounding box (in figure inches).
        """

        bb = []
        if bbox_extra_artists is None:
            artists = self.get_default_bbox_extra_artists()
        else:
            artists = bbox_extra_artists

        for a in artists:
            bbox = a.get_tightbbox(renderer)
            if bbox is not None and (bbox.width != 0 or bbox.height != 0):
                bb.append(bbox)

        bb.extend(
            ax.get_tightbbox(renderer, bbox_extra_artists=bbox_extra_artists)
            for ax in self.axes if ax.get_visible())

        if len(bb) == 0:
            return self.bbox_inches

        _bbox = Bbox.union([b for b in bb if b.width != 0 or b.height != 0])

        bbox_inches = TransformedBbox(_bbox,
                                      Affine2D().scale(1. / self.dpi))

        return bbox_inches

    def init_layoutbox(self):
        """Initialize the layoutbox for use in constrained_layout."""
        if self._layoutbox is None:
            self._layoutbox = layoutbox.LayoutBox(parent=None,
                                     name='figlb',
                                     artist=self)
            self._layoutbox.constrain_geometry(0., 0., 1., 1.)

    def execute_constrained_layout(self, renderer=None):
        """
        Use ``layoutbox`` to determine pos positions within axes.

        See also `.set_constrained_layout_pads`.
        """

        from matplotlib._constrained_layout import do_constrained_layout

        _log.debug('Executing constrainedlayout')
        if self._layoutbox is None:
            warnings.warn("Calling figure.constrained_layout, but figure not "
                          "setup to do constrained layout.  You either called "
                          "GridSpec without the fig keyword, you are using "
                          "plt.subplot, or you need to call figure or "
                          "subplots with the constrained_layout=True kwarg.")
            return
        w_pad, h_pad, wspace, hspace = self.get_constrained_layout_pads()
        # convert to unit-relative lengths
        fig = self
        width, height = fig.get_size_inches()
        w_pad = w_pad / width
        h_pad = h_pad / height
        if renderer is None:
            renderer = layoutbox.get_renderer(fig)
        do_constrained_layout(fig, renderer, h_pad, w_pad, hspace, wspace)

    def tight_layout(self, renderer=None, pad=1.08, h_pad=None, w_pad=None,
                     rect=None):
        """
        Automatically adjust subplot parameters to give specified padding.

        To exclude an artist on the axes from the bounding box calculation
        that determines the subplot parameters (i.e. legend, or annotation),
        then set `a.set_in_layout(False)` for that artist.

        Parameters
        ----------
        renderer : subclass of `~.backend_bases.RendererBase`, optional
            Defaults to the renderer for the figure.

        pad : float, optional
            Padding between the figure edge and the edges of subplots,
            as a fraction of the font size.
        h_pad, w_pad : float, optional
            Padding (height/width) between edges of adjacent subplots,
            as a fraction of the font size.  Defaults to *pad*.
        rect : tuple (left, bottom, right, top), optional
            A rectangle (left, bottom, right, top) in the normalized
            figure coordinate that the whole subplots area (including
            labels) will fit into. Default is (0, 0, 1, 1).

        See Also
        --------
        .Figure.set_tight_layout
        .pyplot.tight_layout
        """

        from .tight_layout import (
            get_renderer, get_subplotspec_list, get_tight_layout_figure)

        subplotspec_list = get_subplotspec_list(self.axes)
        if None in subplotspec_list:
            warnings.warn("This figure includes Axes that are not compatible "
                          "with tight_layout, so results might be incorrect.")

        if renderer is None:
            renderer = get_renderer(self)

        kwargs = get_tight_layout_figure(
            self, self.axes, subplotspec_list, renderer,
            pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        if kwargs:
            self.subplots_adjust(**kwargs)

    def align_xlabels(self, axs=None):
        """
        Align the ylabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the bottom, it is aligned with labels on axes that
        also have their label on the bottom and that have the same
        bottom-most subplot row.  If the label is on the top,
        it is aligned with labels on axes with the same top-most row.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or ndarray) `~matplotlib.axes.Axes`
            to align the xlabels.
            Default is to align all axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_ylabels

        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with rotated xtick labels::

            fig, axs = plt.subplots(1, 2)
            for tick in axs[0].get_xticklabels():
                tick.set_rotation(55)
            axs[0].set_xlabel('XLabel 0')
            axs[1].set_xlabel('XLabel 1')
            fig.align_xlabels()

        """

        if axs is None:
            axs = self.axes
        axs = np.asarray(axs).ravel()
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_xlabel())
            ss = ax.get_subplotspec()
            nrows, ncols, row0, row1, col0, col1 = ss.get_rows_columns()
            labpo = ax.xaxis.get_label_position()  # top or bottom

            # loop through other axes, and search for label positions
            # that are same as this one, and that share the appropriate
            # row number.
            #  Add to a grouper associated with each axes of sibblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc.xaxis.get_label_position() == labpo:
                    ss = axc.get_subplotspec()
                    nrows, ncols, rowc0, rowc1, colc, col1 = \
                            ss.get_rows_columns()
                    if (labpo == 'bottom' and rowc1 == row1 or
                        labpo == 'top' and rowc0 == row0):
                        # grouper for groups of xlabels to align
                        self._align_xlabel_grp.join(ax, axc)

    def align_ylabels(self, axs=None):
        """
        Align the ylabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the left, it is aligned with labels on axes that
        also have their label on the left and that have the same
        left-most subplot column.  If the label is on the right,
        it is aligned with labels on axes with the same right-most column.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or ndarray) of `~matplotlib.axes.Axes`
            to align the ylabels.
            Default is to align all axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels

        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with large yticks labels::

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(0, 1000, 50))
            axs[0].set_ylabel('YLabel 0')
            axs[1].set_ylabel('YLabel 1')
            fig.align_ylabels()

        """

        if axs is None:
            axs = self.axes
        axs = np.asarray(axs).ravel()
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_ylabel())
            ss = ax.get_subplotspec()
            nrows, ncols, row0, row1, col0, col1 = ss.get_rows_columns()
            same = [ax]
            labpo = ax.yaxis.get_label_position()  # left or right
            # loop through other axes, and search for label positions
            # that are same as this one, and that share the appropriate
            # column number.
            # Add to a list associated with each axes of sibblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc != ax:
                    if axc.yaxis.get_label_position() == labpo:
                        ss = axc.get_subplotspec()
                        nrows, ncols, row0, row1, colc0, colc1 = \
                                ss.get_rows_columns()
                        if (labpo == 'left' and colc0 == col0 or
                            labpo == 'right' and colc1 == col1):
                            # grouper for groups of ylabels to align
                            self._align_ylabel_grp.join(ax, axc)

    def align_labels(self, axs=None):
        """
        Align the xlabels and ylabels of subplots with the same subplots
        row or column (respectively) if label alignment is being
        done automatically (i.e. the label position is not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or ndarray) of `~matplotlib.axes.Axes`
            to align the labels.
            Default is to align all axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels

        matplotlib.figure.Figure.align_ylabels
        """
        self.align_xlabels(axs=axs)
        self.align_ylabels(axs=axs)

    def add_gridspec(self, nrows, ncols, **kwargs):
        """
        Return a `.GridSpec` that has this figure as a parent.  This allows
        complex layout of axes in the figure.

        Parameters
        ----------
        nrows : int
            Number of rows in grid.

        ncols : int
            Number or columns in grid.

        Returns
        -------
        gridspec : `.GridSpec`

        Other Parameters
        ----------------
        *kwargs* are passed to `.GridSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding a subplot that spans two rows::

            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            # spans two rows:
            ax3 = fig.add_subplot(gs[:, 1])

        """

        _ = kwargs.pop('figure', None)  # pop in case user has added this...
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self, **kwargs)
        self._gridspecs.append(gs)
        return gs


def figaspect(arg):
    """
    Calculate the width and height for a figure with a specified aspect ratio.

    While the height is taken from :rc:`figure.figsize`, the width is
    adjusted to match the desired aspect ratio. Additionally, it is ensured
    that the width is in the range [4., 16.] and the height is in the range
    [2., 16.]. If necessary, the default height is adjusted to ensure this.

    Parameters
    ----------
    arg : scalar or 2d array
        If a scalar, this defines the aspect ratio (i.e. the ratio height /
        width).
        In case of an array the aspect ratio is number of rows / number of
        columns, so that the array could be fitted in the figure undistorted.

    Returns
    -------
    width, height
        The figure size in inches.

    Notes
    -----
    If you want to create an axes within the figure, that still preserves the
    aspect ratio, be sure to create it with equal width and height. See
    examples below.

    Thanks to Fernando Perez for this function.

    Examples
    --------
    Make a figure twice as tall as it is wide::

        w, h = figaspect(2.)
        fig = Figure(figsize=(w, h))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(A, **kwargs)

    Make a figure with the proper aspect for an array::

        A = rand(5,3)
        w, h = figaspect(A)
        fig = Figure(figsize=(w, h))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(A, **kwargs)
    """

    isarray = hasattr(arg, 'shape') and not np.isscalar(arg)

    # min/max sizes to respect when autoscaling.  If John likes the idea, they
    # could become rc parameters, for now they're hardwired.
    figsize_min = np.array((4.0, 2.0))  # min length for width/height
    figsize_max = np.array((16.0, 16.0))  # max length for width/height

    # Extract the aspect ratio of the array
    if isarray:
        nr, nc = arg.shape[:2]
        arr_ratio = nr / nc
    else:
        arr_ratio = arg

    # Height of user figure defaults
    fig_height = rcParams['figure.figsize'][1]

    # New size for the figure, keeping the aspect ratio of the caller
    newsize = np.array((fig_height / arr_ratio, fig_height))

    # Sanity checks, don't drop either dimension below figsize_min
    newsize /= min(1.0, *(newsize / figsize_min))

    # Avoid humongous windows as well
    newsize /= max(1.0, *(newsize / figsize_max))

    # Finally, if we have a really funky aspect ratio, break it but respect
    # the min/max dimensions (we don't want figures 10 feet tall!)
    newsize = np.clip(newsize, figsize_min, figsize_max)
    return newsize

docstring.interpd.update(Figure=martist.kwdoc(Figure))
