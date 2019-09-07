import functools
import uuid

from matplotlib import cbook, docstring
import matplotlib.artist as martist
from matplotlib.axes._axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec
import matplotlib._layoutbox as layoutbox


class SubplotBase:
    """
    Base class for subplots, which are :class:`Axes` instances with
    additional methods to facilitate generating and manipulating a set
    of :class:`Axes` within a figure.
    """

    def __init__(self, fig, *args, **kwargs):
        """
        Parameters
        ----------
        fig : `matplotlib.figure.Figure`

        *args : tuple (*nrows*, *ncols*, *index*) or int
            The array of subplots in the figure has dimensions ``(nrows,
            ncols)``, and *index* is the index of the subplot being created.
            *index* starts at 1 in the upper left corner and increases to the
            right.

            If *nrows*, *ncols*, and *index* are all single digit numbers, then
            *args* can be passed as a single 3-digit number (e.g. 234 for
            (2, 3, 4)).
        """

        self.figure = fig

        if len(args) == 1:
            if isinstance(args[0], SubplotSpec):
                self._subplotspec = args[0]
            else:
                try:
                    s = str(int(args[0]))
                    rows, cols, num = map(int, s)
                except ValueError:
                    raise ValueError('Single argument to subplot must be '
                        'a 3-digit integer')
                self._subplotspec = GridSpec(rows, cols,
                                             figure=self.figure)[num - 1]
                # num - 1 for converting from MATLAB to python indexing
        elif len(args) == 3:
            rows, cols, num = args
            rows = int(rows)
            cols = int(cols)
            if rows <= 0:
                raise ValueError(f'Number of rows must be > 0, not {rows}')
            if cols <= 0:
                raise ValueError(f'Number of columns must be > 0, not {cols}')
            if isinstance(num, tuple) and len(num) == 2:
                num = [int(n) for n in num]
                self._subplotspec = GridSpec(
                        rows, cols,
                        figure=self.figure)[(num[0] - 1):num[1]]
            else:
                if num < 1 or num > rows*cols:
                    raise ValueError(
                        f"num must be 1 <= num <= {rows*cols}, not {num}")
                self._subplotspec = GridSpec(
                        rows, cols, figure=self.figure)[int(num) - 1]
                # num - 1 for converting from MATLAB to python indexing
        else:
            raise ValueError(f'Illegal argument(s) to subplot: {args}')

        self.update_params()

        # _axes_class is set in the subplot_class_factory
        self._axes_class.__init__(self, fig, self.figbox, **kwargs)
        # add a layout box to this, for both the full axis, and the poss
        # of the axis.  We need both because the axes may become smaller
        # due to parasitic axes and hence no longer fill the subplotspec.
        if self._subplotspec._layoutbox is None:
            self._layoutbox = None
            self._poslayoutbox = None
        else:
            name = self._subplotspec._layoutbox.name + '.ax'
            name = name + layoutbox.seq_id()
            self._layoutbox = layoutbox.LayoutBox(
                    parent=self._subplotspec._layoutbox,
                    name=name,
                    artist=self)
            self._poslayoutbox = layoutbox.LayoutBox(
                    parent=self._layoutbox,
                    name=self._layoutbox.name+'.pos',
                    pos=True, subplot=True, artist=self)

    def __reduce__(self):
        # get the first axes class which does not inherit from a subplotbase
        axes_class = next(
            c for c in type(self).__mro__
            if issubclass(c, Axes) and not issubclass(c, SubplotBase))
        return (_picklable_subplot_class_constructor,
                (axes_class,),
                self.__getstate__())

    def get_geometry(self):
        """Get the subplot geometry, e.g., (2, 2, 3)."""
        rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
        return rows, cols, num1 + 1  # for compatibility

    # COVERAGE NOTE: Never used internally or from examples
    def change_geometry(self, numrows, numcols, num):
        """Change subplot geometry, e.g., from (1, 1, 1) to (2, 2, 3)."""
        self._subplotspec = GridSpec(numrows, numcols,
                                     figure=self.figure)[num - 1]
        self.update_params()
        self.set_position(self.figbox)

    def get_subplotspec(self):
        """get the SubplotSpec instance associated with the subplot"""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """set the SubplotSpec instance associated with the subplot"""
        self._subplotspec = subplotspec

    def get_gridspec(self):
        """get the GridSpec instance associated with the subplot"""
        return self._subplotspec.get_gridspec()

    def update_params(self):
        """update the subplot position from fig.subplotpars"""
        self.figbox, _, _, self.numRows, self.numCols = \
            self.get_subplotspec().get_position(self.figure,
                                                return_all=True)

    @cbook.deprecated("3.2", alternative="ax.get_subplotspec().rowspan.start")
    def rowNum(self):
        return self.get_subplotspec().rowspan.start

    @cbook.deprecated("3.2", alternative="ax.get_subplotspec().colspan.start")
    def colNum(self):
        return self.get_subplotspec().colspan.start

    def is_first_row(self):
        return self.get_subplotspec().rowspan.start == 0

    def is_last_row(self):
        return self.get_subplotspec().rowspan.stop == self.get_gridspec().nrows

    def is_first_col(self):
        return self.get_subplotspec().colspan.start == 0

    def is_last_col(self):
        return self.get_subplotspec().colspan.stop == self.get_gridspec().ncols

    def label_outer(self):
        """
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row; y-labels only for
        subplots on the first column.
        """
        lastrow = self.is_last_row()
        firstcol = self.is_first_col()
        if not lastrow:
            for label in self.get_xticklabels(which="both"):
                label.set_visible(False)
            self.get_xaxis().get_offset_text().set_visible(False)
            self.set_xlabel("")
        if not firstcol:
            for label in self.get_yticklabels(which="both"):
                label.set_visible(False)
            self.get_yaxis().get_offset_text().set_visible(False)
            self.set_ylabel("")

    def _make_twin_axes(self, *args, **kwargs):
        """Make a twinx axes of self. This is used for twinx and twiny."""
        if 'sharex' in kwargs and 'sharey' in kwargs:
            # The following line is added in v2.2 to avoid breaking Seaborn,
            # which currently uses this internal API.
            if kwargs["sharex"] is not self and kwargs["sharey"] is not self:
                raise ValueError("Twinned Axes may share only one axis")
        # The dance here with label is to force add_subplot() to create a new
        # Axes (by passing in a label never seen before).  Note that this does
        # not affect plot reactivation by subplot() as twin axes can never be
        # reactivated by subplot().
        sentinel = str(uuid.uuid4())
        real_label = kwargs.pop("label", sentinel)
        twin = self.figure.add_subplot(
            self.get_subplotspec(), *args, label=sentinel, **kwargs)
        if real_label is not sentinel:
            twin.set_label(real_label)
        self.set_adjustable('datalim')
        twin.set_adjustable('datalim')
        if self._layoutbox is not None and twin._layoutbox is not None:
            # make the layout boxes be explicitly the same
            twin._layoutbox.constrain_same(self._layoutbox)
            twin._poslayoutbox.constrain_same(self._poslayoutbox)
        self._twinned_axes.join(self, twin)
        return twin


# this here to support cartopy which was using a private part of the
# API to register their Axes subclasses.

# In 3.1 this should be changed to a dict subclass that warns on use
# In 3.3 to a dict subclass that raises a useful exception on use
# In 3.4 should be removed

# The slow timeline is to give cartopy enough time to get several
# release out before we break them.
_subplot_classes = {}


@functools.lru_cache(None)
def subplot_class_factory(axes_class=None):
    """
    This makes a new class that inherits from `.SubplotBase` and the
    given axes_class (which is assumed to be a subclass of `.axes.Axes`).
    This is perhaps a little bit roundabout to make a new class on
    the fly like this, but it means that a new Subplot class does
    not have to be created for every type of Axes.
    """
    if axes_class is None:
        axes_class = Axes
    try:
        # Avoid creating two different instances of GeoAxesSubplot...
        # Only a temporary backcompat fix.  This should be removed in
        # 3.4
        return next(cls for cls in SubplotBase.__subclasses__()
                    if cls.__bases__ == (SubplotBase, axes_class))
    except StopIteration:
        return type("%sSubplot" % axes_class.__name__,
                    (SubplotBase, axes_class),
                    {'_axes_class': axes_class})


# This is provided for backward compatibility
Subplot = subplot_class_factory()


def _picklable_subplot_class_constructor(axes_class):
    """
    This stub class exists to return the appropriate subplot class when called
    with an axes class. This is purely to allow pickling of Axes and Subplots.
    """
    subplot_class = subplot_class_factory(axes_class)
    return subplot_class.__new__(subplot_class)


docstring.interpd.update(Axes=martist.kwdoc(Axes))
docstring.dedent_interpd(Axes.__init__)

docstring.interpd.update(Subplot=martist.kwdoc(Axes))
