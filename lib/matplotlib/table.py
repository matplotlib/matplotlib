# Original code by:
#    John Gill <jng@europe.renre.com>
#    Copyright 2004 John Gill and John Hunter
#
# Subsequent changes:
#    The Matplotlib development team
#    Copyright The Matplotlib development team

"""
This module provides functionality to add a table to a plot.

Use the factory function `~matplotlib.table.table` to create a ready-made
table from texts. If you need more control, use the `.Table` class and its
methods.

The table consists of a grid of cells, which are indexed by (row, column).
The cell (0, 0) is positioned at the top left.

Thanks to John Gill for providing the class and table.
"""

from . import artist, cbook, docstring
from .artist import Artist, allow_rasterization
from .patches import Rectangle
from .text import Text
from .transforms import Bbox
from .path import Path


class Cell(Rectangle):
    """
    A cell is a `.Rectangle` with some associated `.Text`.

    .. note:
        As a user, you'll most likely not creates cells yourself. Instead, you
        should use either the `~matplotlib.table.table` factory function or
        `.Table.add_cell`.

    Parameters
    ----------
    xy : 2-tuple
        The position of the bottom left corner of the cell.
    width : float
        The cell width.
    height : float
        The cell height.
    edgecolor : color spec
        The color of the cell border.
    facecolor : color spec
        The cell facecolor.
    fill : bool
        Whether the cell background is filled.
    text : str
        The cell text.
    loc : {'left', 'center', 'right'}, default: 'right'
        The alignment of the text within the cell.
    fontproperties : dict
        A dict defining the font properties of the text. Supported keys and
        values are the keyword arguments accepted by `.FontProperties`.
    """

    PAD = 0.1
    """Padding between text and rectangle."""

    def __init__(self, xy, width, height,
                 edgecolor='k', facecolor='w',
                 fill=True,
                 text='',
                 loc=None,
                 fontproperties=None
                 ):

        # Call base
        Rectangle.__init__(self, xy, width=width, height=height, fill=fill,
                           edgecolor=edgecolor, facecolor=facecolor)
        self.set_clip_on(False)

        # Create text object
        if loc is None:
            loc = 'right'
        self._loc = loc
        self._text = Text(x=xy[0], y=xy[1], text=text,
                          fontproperties=fontproperties)
        self._text.set_clip_on(False)

    def set_transform(self, trans):
        Rectangle.set_transform(self, trans)
        # the text does not get the transform!
        self.stale = True

    def set_figure(self, fig):
        Rectangle.set_figure(self, fig)
        self._text.set_figure(fig)

    def get_text(self):
        """Return the cell `.Text` instance."""
        return self._text

    def set_fontsize(self, size):
        """Set the text fontsize."""
        self._text.set_fontsize(size)
        self.stale = True

    def get_fontsize(self):
        """Return the cell fontsize."""
        return self._text.get_fontsize()

    def auto_set_font_size(self, renderer):
        """Shrink font size until the text fits into the cell width."""
        fontsize = self.get_fontsize()
        required = self.get_required_width(renderer)
        while fontsize > 1 and required > self.get_width():
            fontsize -= 1
            self.set_fontsize(fontsize)
            required = self.get_required_width(renderer)

        return fontsize

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        # draw the rectangle
        Rectangle.draw(self, renderer)

        # position the text
        self._set_text_position(renderer)
        self._text.draw(renderer)
        self.stale = False

    def _set_text_position(self, renderer):
        """Set text up so it draws in the right place.

        Currently support 'left', 'center' and 'right'
        """
        bbox = self.get_window_extent(renderer)
        l, b, w, h = bbox.bounds

        # draw in center vertically
        self._text.set_verticalalignment('center')
        y = b + (h / 2.0)

        # now position horizontally
        if self._loc == 'center':
            self._text.set_horizontalalignment('center')
            x = l + (w / 2.0)
        elif self._loc == 'left':
            self._text.set_horizontalalignment('left')
            x = l + (w * self.PAD)
        else:
            self._text.set_horizontalalignment('right')
            x = l + (w * (1.0 - self.PAD))

        self._text.set_position((x, y))

    def get_text_bounds(self, renderer):
        """
        Return the text bounds as *(x, y, width, height)* in table coordinates.
        """
        bbox = self._text.get_window_extent(renderer)
        bboxa = bbox.inverse_transformed(self.get_data_transform())
        return bboxa.bounds

    def get_required_width(self, renderer):
        """Return the minimal required width for the cell."""
        l, b, w, h = self.get_text_bounds(renderer)
        return w * (1.0 + (2.0 * self.PAD))

    @docstring.dedent_interpd
    def set_text_props(self, **kwargs):
        """
        Update the text properties.

        Valid kwargs are
        %(Text)s
        """
        self._text.update(kwargs)
        self.stale = True


class CustomCell(Cell):
    """
    A `.Cell` subclass with configurable edge visibility.
    """

    _edges = 'BRTL'
    _edge_aliases = {'open':         '',
                     'closed':       _edges,  # default
                     'horizontal':   'BT',
                     'vertical':     'RL'
                     }

    def __init__(self, *args, visible_edges, **kwargs):
        super().__init__(*args, **kwargs)
        self.visible_edges = visible_edges

    @property
    def visible_edges(self):
        """
        The cell edges to be drawn with a line.

        Reading this property returns a substring of 'BRTL' (bottom, right,
        top, left').

        When setting this property, you can use a substring of 'BRTL' or one
        of {'open', 'closed', 'horizontal', 'vertical'}.
        """
        return self._visible_edges

    @visible_edges.setter
    def visible_edges(self, value):
        if value is None:
            self._visible_edges = self._edges
        elif value in self._edge_aliases:
            self._visible_edges = self._edge_aliases[value]
        else:
            if any(edge not in self._edges for edge in value):
                raise ValueError('Invalid edge param {}, must only be one of '
                                 '{} or string of {}'.format(
                                     value,
                                     ", ".join(self._edge_aliases),
                                     ", ".join(self._edges)))
            self._visible_edges = value
        self.stale = True

    def get_path(self):
        """Return a `.Path` for the `.visible_edges`."""
        codes = [Path.MOVETO]

        for edge in self._edges:
            if edge in self._visible_edges:
                codes.append(Path.LINETO)
            else:
                codes.append(Path.MOVETO)

        if Path.MOVETO not in codes[1:]:  # All sides are visible
            codes[-1] = Path.CLOSEPOLY

        return Path(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            codes,
            readonly=True
            )


class Table(Artist):
    """
    A table of cells.

    The table consists of a grid of cells, which are indexed by (row, column).

    For a simple table, you'll have a full grid of cells with indices from
    (0, 0) to (num_rows-1, num_cols-1), in which the cell (0, 0) is positioned
    at the top left. However, you can also add cells with negative indices.
    You don't have to add a cell to every grid position, so you can create
    tables that have holes.

    *Note*: You'll usually not create an empty table from scratch. Instead use
    `~matplotlib.table.table` to create a table from data.
    """
    codes = {'best': 0,
             'upper right':  1,  # default
             'upper left':   2,
             'lower left':   3,
             'lower right':  4,
             'center left':  5,
             'center right': 6,
             'lower center': 7,
             'upper center': 8,
             'center':       9,
             'top right':    10,
             'top left':     11,
             'bottom left':  12,
             'bottom right': 13,
             'right':        14,
             'left':         15,
             'top':          16,
             'bottom':       17,
             }
    """Possible values where to place the table relative to the Axes."""

    FONTSIZE = 10

    AXESPAD = 0.02
    """The border between the Axes and the table edge in Axes units."""

    def __init__(self, ax, loc=None, bbox=None, **kwargs):
        """
        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            The `~.axes.Axes` to plot the table into.
        loc : str
            The position of the cell with respect to *ax*. This must be one of
            the `~.Table.codes`.
        bbox : `.Bbox` or None
            A bounding box to draw the table into. If this is not *None*, this
            overrides *loc*.

        Other Parameters
        ----------------
        **kwargs
            `.Artist` properties.
        """

        Artist.__init__(self)

        if isinstance(loc, str):
            if loc not in self.codes:
                cbook.warn_deprecated(
                    "3.1", message="Unrecognized location {!r}. Falling back "
                    "on 'bottom'; valid locations are\n\t{}\n"
                    "This will raise an exception %(removal)s."
                    .format(loc, '\n\t'.join(self.codes)))
                loc = 'bottom'
            loc = self.codes[loc]
        self.set_figure(ax.figure)
        self._axes = ax
        self._loc = loc
        self._bbox = bbox

        # use axes coords
        self.set_transform(ax.transAxes)

        self._cells = {}
        self._edges = None
        self._autoColumns = []
        self._autoFontsize = True
        self.update(kwargs)

        self.set_clip_on(False)

    def add_cell(self, row, col, *args, **kwargs):
        """
        Create a cell and add it to the table.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        *args, **kwargs
            All other parameters are passed on to `Cell`.

        Returns
        -------
        cell : `.CustomCell`
            The created cell.

        """
        xy = (0, 0)
        cell = CustomCell(xy, visible_edges=self.edges, *args, **kwargs)
        self[row, col] = cell
        return cell

    def __setitem__(self, position, cell):
        """
        Set a custom cell in a given position.
        """
        if not isinstance(cell, CustomCell):
            raise TypeError('Table only accepts CustomCell')
        try:
            row, col = position[0], position[1]
        except Exception:
            raise KeyError('Only tuples length 2 are accepted as coordinates')
        cell.set_figure(self.figure)
        cell.set_transform(self.get_transform())
        cell.set_clip_on(False)
        self._cells[row, col] = cell
        self.stale = True

    def __getitem__(self, position):
        """Retrieve a custom cell from a given position."""
        return self._cells[position]

    @property
    def edges(self):
        """
        The default value of `~.CustomCell.visible_edges` for newly added
        cells using `.add_cell`.

        Notes
        -----
        This setting does currently only affect newly created cells using
        `.add_cell`.

        To change existing cells, you have to set their edges explicitly::

            for c in tab.get_celld().values():
                c.visible_edges = 'horizontal'

        """
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value
        self.stale = True

    def _approx_text_height(self):
        return (self.FONTSIZE / 72.0 * self.figure.dpi /
                self._axes.bbox.height * 1.2)

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        # Need a renderer to do hit tests on mouseevent; assume the last one
        # will do
        if renderer is None:
            renderer = self.figure._cachedRenderer
        if renderer is None:
            raise RuntimeError('No renderer defined')

        if not self.get_visible():
            return
        renderer.open_group('table')
        self._update_positions(renderer)

        for key in sorted(self._cells):
            self._cells[key].draw(renderer)

        renderer.close_group('table')
        self.stale = False

    def _get_grid_bbox(self, renderer):
        """Get a bbox, in axes co-ordinates for the cells.

        Only include those in the range (0,0) to (maxRow, maxCol)"""
        boxes = [cell.get_window_extent(renderer)
                 for (row, col), cell in self._cells.items()
                 if row >= 0 and col >= 0]
        bbox = Bbox.union(boxes)
        return bbox.inverse_transformed(self.get_transform())

    def contains(self, mouseevent):
        # docstring inherited
        if self._contains is not None:
            return self._contains(self, mouseevent)

        # TODO: Return index of the cell containing the cursor so that the user
        # doesn't have to bind to each one individually.
        renderer = self.figure._cachedRenderer
        if renderer is not None:
            boxes = [cell.get_window_extent(renderer)
                     for (row, col), cell in self._cells.items()
                     if row >= 0 and col >= 0]
            bbox = Bbox.union(boxes)
            return bbox.contains(mouseevent.x, mouseevent.y), {}
        else:
            return False, {}

    def get_children(self):
        """Return the Artists contained by the table."""
        return list(self._cells.values())
    get_child_artists = cbook.deprecated("3.0")(get_children)

    def get_window_extent(self, renderer):
        """Return the bounding box of the table in window coords."""
        boxes = [cell.get_window_extent(renderer)
                 for cell in self._cells.values()]
        return Bbox.union(boxes)

    def _do_cell_alignment(self):
        """
        Calculate row heights and column widths; position cells accordingly.
        """
        # Calculate row/column widths
        widths = {}
        heights = {}
        for (row, col), cell in self._cells.items():
            height = heights.setdefault(row, 0.0)
            heights[row] = max(height, cell.get_height())
            width = widths.setdefault(col, 0.0)
            widths[col] = max(width, cell.get_width())

        # work out left position for each column
        xpos = 0
        lefts = {}
        for col in sorted(widths):
            lefts[col] = xpos
            xpos += widths[col]

        ypos = 0
        bottoms = {}
        for row in sorted(heights, reverse=True):
            bottoms[row] = ypos
            ypos += heights[row]

        # set cell positions
        for (row, col), cell in self._cells.items():
            cell.set_x(lefts[col])
            cell.set_y(bottoms[row])

    def auto_set_column_width(self, col):
        """
        Automatically set the widths of given columns to optimal sizes.

        Parameters
        ----------
        col : int or sequence of ints
            The indices of the columns to auto-scale.
        """
        # check for col possibility on iteration
        try:
            iter(col)
        except (TypeError, AttributeError):
            self._autoColumns.append(col)
        else:
            for cell in col:
                self._autoColumns.append(cell)

        self.stale = True

    def _auto_set_column_width(self, col, renderer):
        """Automatically set width for column."""
        cells = [cell for key, cell in self._cells.items() if key[1] == col]
        max_width = max((cell.get_required_width(renderer) for cell in cells),
                        default=0)
        for cell in cells:
            cell.set_width(max_width)

    def auto_set_font_size(self, value=True):
        """Automatically set font size."""
        self._autoFontsize = value
        self.stale = True

    def _auto_set_font_size(self, renderer):

        if len(self._cells) == 0:
            return
        fontsize = next(iter(self._cells.values())).get_fontsize()
        cells = []
        for key, cell in self._cells.items():
            # ignore auto-sized columns
            if key[1] in self._autoColumns:
                continue
            size = cell.auto_set_font_size(renderer)
            fontsize = min(fontsize, size)
            cells.append(cell)

        # now set all fontsizes equal
        for cell in self._cells.values():
            cell.set_fontsize(fontsize)

    def scale(self, xscale, yscale):
        """Scale column widths by *xscale* and row heights by *yscale*."""
        for c in self._cells.values():
            c.set_width(c.get_width() * xscale)
            c.set_height(c.get_height() * yscale)

    def set_fontsize(self, size):
        """
        Set the font size, in points, of the cell text.

        Parameters
        ----------
        size : float

        Notes
        -----
        As long as auto font size has not been disabled, the value will be
        clipped such that the text fits horizontally into the cell.

        You can disable this behavior using `.auto_set_font_size`.

        >>> the_table.auto_set_font_size(False)
        >>> the_table.set_fontsize(20)

        However, there is no automatic scaling of the row height so that the
        text may exceed the cell boundary.
        """
        for cell in self._cells.values():
            cell.set_fontsize(size)
        self.stale = True

    def _offset(self, ox, oy):
        """Move all the artists by ox, oy (axes coords)."""
        for c in self._cells.values():
            x, y = c.get_x(), c.get_y()
            c.set_x(x + ox)
            c.set_y(y + oy)

    def _update_positions(self, renderer):
        # called from renderer to allow more precise estimates of
        # widths and heights with get_window_extent

        # Do any auto width setting
        for col in self._autoColumns:
            self._auto_set_column_width(col, renderer)

        if self._autoFontsize:
            self._auto_set_font_size(renderer)

        # Align all the cells
        self._do_cell_alignment()

        bbox = self._get_grid_bbox(renderer)
        l, b, w, h = bbox.bounds

        if self._bbox is not None:
            # Position according to bbox
            rl, rb, rw, rh = self._bbox
            self.scale(rw / w, rh / h)
            ox = rl - l
            oy = rb - b
            self._do_cell_alignment()
        else:
            # Position using loc
            (BEST, UR, UL, LL, LR, CL, CR, LC, UC, C,
             TR, TL, BL, BR, R, L, T, B) = range(len(self.codes))
            # defaults for center
            ox = (0.5 - w / 2) - l
            oy = (0.5 - h / 2) - b
            if self._loc in (UL, LL, CL):   # left
                ox = self.AXESPAD - l
            if self._loc in (BEST, UR, LR, R, CR):  # right
                ox = 1 - (l + w + self.AXESPAD)
            if self._loc in (BEST, UR, UL, UC):     # upper
                oy = 1 - (b + h + self.AXESPAD)
            if self._loc in (LL, LR, LC):           # lower
                oy = self.AXESPAD - b
            if self._loc in (LC, UC, C):            # center x
                ox = (0.5 - w / 2) - l
            if self._loc in (CL, CR, C):            # center y
                oy = (0.5 - h / 2) - b

            if self._loc in (TL, BL, L):            # out left
                ox = - (l + w)
            if self._loc in (TR, BR, R):            # out right
                ox = 1.0 - l
            if self._loc in (TR, TL, T):            # out top
                oy = 1.0 - b
            if self._loc in (BL, BR, B):           # out bottom
                oy = - (b + h)

        self._offset(ox, oy)

    def get_celld(self):
        r"""
        Return a dict of cells in the table mapping *(row, column)* to
        `.Cell`\s.

        Notes
        -----
        You can also directly index into the Table object to access individual
        cells::

            cell = table[row, col]

        """
        return self._cells


docstring.interpd.update(Table=artist.kwdoc(Table))


@docstring.dedent_interpd
def table(ax,
          cellText=None, cellColours=None,
          cellLoc='right', colWidths=None,
          rowLabels=None, rowColours=None, rowLoc='left',
          colLabels=None, colColours=None, colLoc='center',
          loc='bottom', bbox=None, edges='closed',
          **kwargs):
    """
    Add a table to an `~.axes.Axes`.

    At least one of *cellText* or *cellColours* must be specified. These
    parameters must be 2D lists, in which the outer lists define the rows and
    the inner list define the column values per row. Each row must have the
    same number of elements.

    The table can optionally have row and column headers, which are configured
    using *rowLabels*, *rowColours*, *rowLoc* and *colLabels*, *colColours*,
    *colLoc* respectively.

    For finer grained control over tables, use the `.Table` class and add it to
    the axes with `.Axes.add_table`.

    Parameters
    ----------
    cellText : 2D list of str, optional
        The texts to place into the table cells.

        *Note*: Line breaks in the strings are currently not accounted for and
        will result in the text exceeding the cell boundaries.

    cellColours : 2D list of matplotlib color specs, optional
        The background colors of the cells.

    cellLoc : {'left', 'center', 'right'}, default: 'right'
        The alignment of the text within the cells.

    colWidths : list of float, optional
        The column widths in units of the axes. If not given, all columns will
        have a width of *1 / ncols*.

    rowLabels : list of str, optional
        The text of the row header cells.

    rowColours : list of matplotlib color specs, optional
        The colors of the row header cells.

    rowLoc : {'left', 'center', 'right'}, optional, default: 'left'
        The text alignment of the row header cells.

    colLabels : list of str, optional
        The text of the column header cells.

    colColours : list of matplotlib color specs, optional
        The colors of the column header cells.

    rowLoc : {'left', 'center', 'right'}, optional, default: 'left'
        The text alignment of the column header cells.

    loc : str, optional
        The position of the cell with respect to *ax*. This must be one of
        the `~.Table.codes`.

    bbox : `.Bbox`, optional
        A bounding box to draw the table into. If this is not *None*, this
        overrides *loc*.

    edges : substring of 'BRTL' or {'open', 'closed', 'horizontal', 'vertical'}
        The cell edges to be drawn with a line. See also
        `~.CustomCell.visible_edges`.

    Other Parameters
    ----------------
    **kwargs
        `.Table` properties.

    %(Table)s

    Returns
    -------
    table : `~matplotlib.table.Table`
        The created table.
    """

    if cellColours is None and cellText is None:
        raise ValueError('At least one argument from "cellColours" or '
                         '"cellText" must be provided to create a table.')

    # Check we have some cellText
    if cellText is None:
        # assume just colours are needed
        rows = len(cellColours)
        cols = len(cellColours[0])
        cellText = [[''] * cols] * rows

    rows = len(cellText)
    cols = len(cellText[0])
    for row in cellText:
        if len(row) != cols:
            raise ValueError("Each row in 'cellText' must have {} columns"
                             .format(cols))

    if cellColours is not None:
        if len(cellColours) != rows:
            raise ValueError("'cellColours' must have {} rows".format(rows))
        for row in cellColours:
            if len(row) != cols:
                raise ValueError("Each row in 'cellColours' must have {} "
                                 "columns".format(cols))
    else:
        cellColours = ['w' * cols] * rows

    # Set colwidths if not given
    if colWidths is None:
        colWidths = [1.0 / cols] * cols

    # Fill in missing information for column
    # and row labels
    rowLabelWidth = 0
    if rowLabels is None:
        if rowColours is not None:
            rowLabels = [''] * rows
            rowLabelWidth = colWidths[0]
    elif rowColours is None:
        rowColours = 'w' * rows

    if rowLabels is not None:
        if len(rowLabels) != rows:
            raise ValueError("'rowLabels' must be of length {0}".format(rows))

    # If we have column labels, need to shift
    # the text and colour arrays down 1 row
    offset = 1
    if colLabels is None:
        if colColours is not None:
            colLabels = [''] * cols
        else:
            offset = 0
    elif colColours is None:
        colColours = 'w' * cols

    # Set up cell colours if not given
    if cellColours is None:
        cellColours = ['w' * cols] * rows

    # Now create the table
    table = Table(ax, loc, bbox, **kwargs)
    table.edges = edges
    height = table._approx_text_height()

    # Add the cells
    for row in range(rows):
        for col in range(cols):
            table.add_cell(row + offset, col,
                           width=colWidths[col], height=height,
                           text=cellText[row][col],
                           facecolor=cellColours[row][col],
                           loc=cellLoc)
    # Do column labels
    if colLabels is not None:
        for col in range(cols):
            table.add_cell(0, col,
                           width=colWidths[col], height=height,
                           text=colLabels[col], facecolor=colColours[col],
                           loc=colLoc)

    # Do row labels
    if rowLabels is not None:
        for row in range(rows):
            table.add_cell(row + offset, -1,
                           width=rowLabelWidth or 1e-15, height=height,
                           text=rowLabels[row], facecolor=rowColours[row],
                           loc=rowLoc)
        if rowLabelWidth == 0:
            table.auto_set_column_width(-1)

    ax.add_table(table)
    return table
