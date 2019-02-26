import matplotlib.cbook as cbook
import matplotlib.artist as martist


class Container(tuple):
    """
    Base class for containers.

    Containers are classes that collect semantically related Artists such as
    the bars of a bar plot.
    """

    def __repr__(self):
        return ("<{} object of {} artists>"
                .format(type(self).__name__, len(self)))

    def __new__(cls, *args, **kwargs):
        return tuple.__new__(cls, args[0])

    def __init__(self, kl, label=None):

        self.eventson = False  # fire events only if eventson
        self._oid = 0  # an observer id
        self._propobservers = {}  # a dict from oids to funcs

        self._remove_method = None

        self.set_label(label)

    @cbook.deprecated("3.0")
    def set_remove_method(self, f):
        self._remove_method = f

    def remove(self):
        for c in cbook.flatten(
                self, scalarp=lambda x: isinstance(x, martist.Artist)):
            if c is not None:
                c.remove()

        if self._remove_method:
            self._remove_method(self)

    def get_label(self):
        """
        Get the label used for this artist in the legend.
        """
        return self._label

    def set_label(self, s):
        """
        Set the label to *s* for auto legend.

        Parameters
        ----------
        s : object
            Any object other than None gets converted to its `str`.
        """
        if s is not None:
            self._label = str(s)
        else:
            self._label = None
        self.pchanged()

    def add_callback(self, func):
        """
        Adds a callback function that will be called whenever one of
        the :class:`Artist`'s properties changes.

        Returns an *id* that is useful for removing the callback with
        :meth:`remove_callback` later.
        """
        oid = self._oid
        self._propobservers[oid] = func
        self._oid += 1
        return oid

    def remove_callback(self, oid):
        """
        Remove a callback based on its *id*.

        .. seealso::

            :meth:`add_callback`
               For adding callbacks

        """
        try:
            del self._propobservers[oid]
        except KeyError:
            pass

    def pchanged(self):
        """
        Fire an event when property changed, calling all of the
        registered callbacks.
        """
        for oid, func in list(self._propobservers.items()):
            func(self)

    def get_children(self):
        return [child for child in cbook.flatten(self) if child is not None]


class BarContainer(Container):
    """
    Container for the artists of bar plots (e.g. created by `.Axes.bar`).

    The container can be treated as a tuple of the *patches* themselves.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    patches : list of :class:`~matplotlib.patches.Rectangle`
        The artists of the bars.

    errorbar : None or :class:`~matplotlib.container.ErrorbarContainer`
        A container for the error bar artists if error bars are present.
        *None* otherwise.

    """

    def __init__(self, patches, errorbar=None, **kwargs):
        self.patches = patches
        self.errorbar = errorbar
        Container.__init__(self, patches, **kwargs)


class ErrorbarContainer(Container):
    """
    Container for the artists of error bars (e.g. created by `.Axes.errorbar`).

    The container can be treated as the *lines* tuple itself.
    Additionally, you can access these and further parameters by the
    attributes.

    Attributes
    ----------
    lines : tuple
        Tuple of ``(data_line, caplines, barlinecols)``.

        - data_line : :class:`~matplotlib.lines.Line2D` instance of
          x, y plot markers and/or line.
        - caplines : tuple of :class:`~matplotlib.lines.Line2D` instances of
          the error bar caps.
        - barlinecols : list of :class:`~matplotlib.collections.LineCollection`
          with the horizontal and vertical error ranges.

    has_xerr, has_yerr : bool
        ``True`` if the errorbar has x/y errors.

    """

    def __init__(self, lines, has_xerr=False, has_yerr=False, **kwargs):
        self.lines = lines
        self.has_xerr = has_xerr
        self.has_yerr = has_yerr
        Container.__init__(self, lines, **kwargs)


class StemContainer(Container):
    """
    Container for the artists created in a :meth:`.Axes.stem` plot.

    The container can be treated like a namedtuple ``(markerline, stemlines,
    baseline)``.

    Attributes
    ----------
    markerline :  :class:`~matplotlib.lines.Line2D`
        The artist of the markers at the stem heads.

    stemlines : list of :class:`~matplotlib.lines.Line2D`
        The artists of the vertical lines for all stems.

    baseline : :class:`~matplotlib.lines.Line2D`
        The artist of the horizontal baseline.
    """
    def __init__(self, markerline_stemlines_baseline, **kwargs):
        """
        Parameters
        ----------
        markerline_stemlines_baseline : tuple
            Tuple of ``(markerline, stemlines, baseline)``.
            ``markerline`` contains the `LineCollection` of the markers,
            ``stemlines`` is a `LineCollection` of the main lines,
            ``baseline`` is the `Line2D` of the baseline.
        """
        markerline, stemlines, baseline = markerline_stemlines_baseline
        self.markerline = markerline
        self.stemlines = stemlines
        self.baseline = baseline
        Container.__init__(self, markerline_stemlines_baseline, **kwargs)
