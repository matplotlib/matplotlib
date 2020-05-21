"""
A layoutgrid is a nrows by ncols set of boxes, meant to be used by
`._constrained_layout`, each box is aalagous to a subplotspec element of
a gridspec.

Each box is defined by left[ncols], right[ncols], bottom[nrow] and top[nrows],
and by an editable margin for each side that gets its value set by
the size of ticklabels, titles, etc on each axes that is in the figure.  The
"inner" widths and heights of these boxes are then constrained to
be the same (relative the values of `width_ratios[ncols]` and
`height_ratios[nrows]`).

The layoutgrid is then constrained to be contained within a parent
layoutgrid, its column(s) and row(s) specified when it is created.
"""

import itertools
import kiwisolver as kiwi
import logging
import numpy as np
from matplotlib.transforms import Bbox


_log = logging.getLogger(__name__)


# renderers can be complicated
def get_renderer(fig):
    if fig._cachedRenderer:
        renderer = fig._cachedRenderer
    else:
        canvas = fig.canvas
        if canvas and hasattr(canvas, "get_renderer"):
            renderer = canvas.get_renderer()
        else:
            # not sure if this can happen
            # seems to with PDF...
            _log.info("constrained_layout : falling back to Agg renderer")
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig)
            renderer = canvas.get_renderer()

    return renderer


class LayoutGrid:
    """
    Analagous to a gridspec, and contained in another LayoutGrid.
    """

    def __init__(self, parent=None, parent_pos=(0, 0),
                 parent_inner=False, name='', ncols=1, nrows=1,
                 h_pad=None, w_pad=None, width_ratios=None,
                 height_ratios=None):
        Variable = kiwi.Variable
        self.parent = parent
        self.parent_pos = parent_pos
        self.parent_inner = parent_inner
        self.name = name
        self.nrows = nrows
        self.ncols = ncols
        self.height_ratios = np.atleast_1d(height_ratios)
        if height_ratios is None:
            self.height_ratios = np.ones(nrows)
        self.width_ratios = np.atleast_1d(width_ratios)
        if width_ratios is None:
            self.width_ratios = np.ones(ncols)

        sn = self.name + '_'
        if parent is None:
            self.parent = None
            self.solver = kiwi.Solver()
        else:
            self.parent = parent
            parent.add_child(self, *parent_pos)
            self.solver = self.parent.solver

        # keep track of artist associated w/ this layout.  Can be none
        self.artists = np.empty((nrows, ncols), dtype=object)
        self.children = np.empty((nrows, ncols), dtype=object)

        self.margins = {}
        self.margin_vals = {}
        # all the boxes in each column share the same left/right margins:
        for todo in ['left', 'right']:
            self.margins[todo] = np.empty((ncols), dtype=object)
            # track the value so we can change only if a margin is larger
            # than the current value
            self.margin_vals[todo] = np.zeros(ncols)

        # These are redundant, but make life easier if
        # we define them all.  All that is really
        # needed is left/right, margin['left'], and margin['right']
        self.widths = np.empty((ncols), dtype=object)
        self.lefts = np.empty((ncols), dtype=object)
        self.rights = np.empty((ncols), dtype=object)
        self.inner_widths = np.empty((ncols), dtype=object)

        # make the variables:
        sol = self.solver
        for i in range(self.ncols):
            for todo in ['left', 'right']:
                self.margins[todo][i] = Variable(f'{sn}margins[{todo}][{i}]')
                sol.addEditVariable(self.margins[todo][i], 'strong')
            self.rights[i] = Variable(f'{sn}rights[{i}]')
            self.lefts[i] = Variable(f'{sn}lefts[{i}]')
            self.widths[i] = Variable(f'{sn}widths[{i}]')
            self.inner_widths[i] = Variable(f'{sn}inner_widths[{i}]')

        for todo in ['bottom', 'top']:
            self.margins[todo] = np.empty((nrows), dtype=object)
            self.margin_vals[todo] = np.zeros(nrows)

        self.heights = np.empty((nrows), dtype=object)
        self.inner_heights = np.empty((nrows), dtype=object)
        self.bottoms = np.empty((nrows), dtype=object)
        self.tops = np.empty((nrows), dtype=object)

        for i in range(self.nrows):
            for todo in ['bottom', 'top']:
                self.margins[todo][i] = Variable(f'{sn}margins[{todo}][{i}]')
                sol.addEditVariable(self.margins[todo][i], 'strong')
            self.bottoms[i] = Variable(f'{sn}bottoms[{i}]')
            self.tops[i] = Variable(f'{sn}tops[{i}]')
            self.inner_heights[i] = Variable(f'{sn}inner_heights[{i}]')
            self.heights[i] = Variable(f'{sn}heights[{i}]')

        # set these margins to zero by default. They will be edited as
        # children are filled.
        self.reset_margins()
        self.add_constraints()

        self.h_pad = h_pad
        self.w_pad = w_pad

    def __repr__(self):
        str = f'LayoutBox: {self.name:25s} {self.nrows}x{self.ncols},\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                str += f'{i}, {j}: '\
                       f'L({self.lefts[j].value():1.3f}, ' \
                       f'B{self.bottoms[i].value():1.3f}, ' \
                       f'W{self.widths[j].value():1.3f}, ' \
                       f'H{self.heights[i].value():1.3f}, ' \
                       f'innerW{self.inner_widths[j].value():1.3f}, ' \
                       f'innerH{self.inner_heights[i].value():1.3f}, ' \
                       f'ML{self.margins["left"][j].value():1.3f}, ' \
                       f'MR{self.margins["right"][j].value():1.3f}, \n'
        return str

    def reset_margins(self):
        """
        Reset all the margins to zero.  Must do this after changing
        figure size, for sinatnce, because the relative size of the
        axes labels etc changes.
        """
        for todo in ['left', 'right', 'bottom', 'top']:
            self.edit_margins(todo, 0.0)

    def add_constraints(self):
        # define self-consistent constraints
        self.hard_constraints()
        # define relationship with parent layoutgrid:
        self.parent_constrain()
        # define relative widths of the grid cells to each other
        # and stack horizontally and vertically.
        self.grid_constraints()

    def hard_constraints(self):
        """
        These are the redundant constraints, plus ones that make the
        rest of the code easier.
        """
        for i in range(self.ncols):
            hc = [self.rights[i] >= self.lefts[i],
                  (self.rights[i] - self.margins['right'][i] >=
                    self.lefts[i] - self.margins['left'][i])]
            for c in hc:
                self.solver.addConstraint(c | 'required')

        for i in range(self.nrows):
            hc = [self.heights[i] == self.tops[i] - self.bottoms[i],
                  self.tops[i] >= self.bottoms[i],
                  (self.tops[i] - self.margins['top'][i] >=
                    self.bottoms[i] - self.margins['bottom'][i])]
            for c in hc:
                self.solver.addConstraint(c | 'required')

    def add_child(self, child, i=0, j=0):
        self.children[i, j] = child

    def parent_constrain(self):
        # constraints that are due to the parent...
        # i.e. the first column's left is equal to the
        # parent's left, the last column right equal to the
        # parent's right...
        parent = self.parent
        if self.parent is None:
            hc = [self.lefts[0] == 0,
                  self.rights[-1] == 1,
                  # top and bottom reversed order...
                  self.tops[0] == 1,
                  self.bottoms[-1] == 0]
        else:
            rows, cols = self.parent_pos
            rows = np.atleast_1d(rows)
            cols = np.atleast_1d(cols)

            left = parent.lefts[cols[0]]
            right = parent.rights[cols[-1]]
            top = parent.tops[rows[0]]
            bottom = parent.bottoms[rows[-1]]
            if self.parent_inner:
                # the layout grid is contained inside the inner
                # grid of the parent.
                left += parent.margins['left'][cols[0]]
                right -= parent.margins['right'][cols[-1]]
                top -= parent.margins['top'][rows[0]]
                bottom += parent.margins['bottom'][rows[-1]]
            hc = [self.lefts[0] == left,
                  self.rights[-1] == right,
                  # from top to bottom
                  self.tops[0] == top,
                  self.bottoms[-1] == bottom]
        for c in hc:
            self.solver.addConstraint(c | 'required')

    def grid_constraints(self):
        # constrain the ratio of the inner part of the grids
        # to be the same (relative to width_ratios)

        # constrain widths:
        iw = self.rights[0] - self.margins['right'][0]
        iw = iw - self.lefts[0] - self.margins['left'][0]
        w0 = iw / self.width_ratios[0]
        # from left to right
        for i in range(1, self.ncols):
            iw = self.rights[i] - self.margins['right'][i]
            iw = iw - self.lefts[i] - self.margins['left'][i]
            w = iw
            c = (w == w0 * self.width_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly next to each other.
            c = (self.rights[i - 1] == self.lefts[i])
            self.solver.addConstraint(c | 'strong')

        # constrain heights:
        ih = self.tops[0] - self.margins['top'][0]
        ih = ih - self.bottoms[0] - self.margins['bottom'][0]
        h0 = ih / self.height_ratios[0]
        # from top to bottom:
        for i in range(1, self.nrows):
            ih = self.tops[i] - self.margins['top'][i]
            h = ih - self.bottoms[i] - self.margins['bottom'][i]
            c = (h == h0 * self.height_ratios[i])
            self.solver.addConstraint(c | 'strong')
            # constrain the grid cells to be directly above each other.
            c = (self.bottoms[i - 1] == self.tops[i])
            self.solver.addConstraint(c | 'strong')

    # Margin editing:  The margins are variable and meant to
    # contain things of a fixes size like axes labels, tick labels, titles
    # etc
    def edit_margin(self, todo, width, col):
        """
        Change the size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        width : float
            Size of the margin.  If it is larger than the existing minimum it
            updates the margin size. Fraction of figure size.

        col : int
            Cell column or row to edit.
        """

        self.solver.suggestValue(self.margins[todo][col], width)
        self.margin_vals[todo][col] = width

    def edit_margin_min(self, todo, width, col=0):
        """
        Change the minimum size of the margin for one cell.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        width : float
            Minimum size of the margin .  If it is larger than the
            existig minimum it updates the margin size. Fraction of
            figure size.

        col: int
            Cell column or row to edit.
        """

        if width > self.margin_vals[todo][col]:
            self.edit_margin(todo, width, col)

    def edit_margins(self, todo, width):
        """
        Change the size of all the margin of all the cells in the layout grid.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        width : float
            Size to set the margins.  Fraction of figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            self.edit_margin(todo, width, i)

    def edit_margins_min(self, todo, width):
        """
        Change the minimum size of all the margin of all
        the cells in the layout grid.

        Parameters
        ----------
        todo: string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        width: float
            Minimum size of the margin .  If it is larger than the
            existig minimum it updates the margin size. Fraction of
            figure size.
        """

        for i in range(len(self.margin_vals[todo])):
            self.edit_margin_min(todo, width, i)

    def edit_outer_margin_mins(self, margin, ss):
        """
        Edit all four margin minimums in one statement.

        Parameters
        ----------
        margin: dict
            size of margins in a dict with keys 'left', 'right', 'bottom',
            'top'

        ss: SubplotSpec
            defines the subplotspec these margins should be applied to
        """
        self.edit_margin_min('left', margin['left'], ss.colspan.start)
        self.edit_margin_min('right', margin['right'], ss.colspan.stop - 1)
        # rows are from the top down:
        self.edit_margin_min('top', margin['top'], ss.rowspan.start)
        self.edit_margin_min('bottom', margin['bottom'], ss.rowspan.stop - 1)

    def get_margins(self, todo, col):
        """Return the margin at this position"""
        return self.margin_vals[todo][col]

    def get_outer_bbox(self, rows=[0], cols=[0]):
        """
        Return the outer bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            self.lefts[cols[0]].value(),
            self.bottoms[rows[-1]].value(),
            self.rights[cols[-1]].value(),
            self.tops[rows[0]].value())
        return bbox

    def get_inner_bbox(self, rows=[0], cols=[0]):
        """
        Return the inner bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value() +
             self.margins['left'][cols[0]].value()),
            (self.bottoms[rows[-1]].value() +
             self.margins['bottom'][rows[-1]].value()),
            (self.rights[cols[-1]].value() -
             self.margins['right'][cols[-1]].value()),
            (self.tops[rows[0]].value() -
             self.margins['top'][rows[0]].value()))
        return bbox

    def get_left_margin_bbox(self, rows=[0], cols=[0]):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value()),
            (self.bottoms[rows[-1]].value()),
            (self.lefts[cols[0]].value() +
                self.margins['left'][cols[0]].value()),
            (self.tops[rows[0]].value()))
        return bbox

    def get_bottom_margin_bbox(self, rows=[0], cols=[0]):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value()),
            (self.bottoms[rows[-1]].value()),
            (self.rights[cols[-1]].value()),
            (self.bottoms[rows[-1]].value() +
                self.margins['bottom'][rows[-1]].value()))
        return bbox

    def get_right_margin_bbox(self, rows=[0], cols=[0]):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.rights[cols[-1]].value() -
                self.margins['right'][cols[-1]].value()),
            (self.bottoms[rows[-1]].value()),
            (self.rights[cols[-1]].value()),
            (self.tops[rows[0]].value()))
        return bbox

    def get_top_margin_bbox(self, rows=[0], cols=[0]):
        """
        Return the left margin bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            (self.lefts[cols[0]].value()),
            (self.tops[rows[0]].value()),
            (self.rights[cols[-1]].value()),
            (self.tops[rows[0]].value() -
                self.margins['top'][rows[0]].value()))
        return bbox

    def update_variables(self):
        """
        Update the variables for the solver attached to this layoutgrid.
        """
        self.solver.updateVariables()

_layoutboxobjnum = itertools.count()


def seq_id():
    """Generate a short sequential id for layoutbox objects."""
    return '%06d' % next(_layoutboxobjnum)


def print_children(lb):
    """Print the children of the layoutbox."""
    for child in lb.children:
        print_children(child)


def nonetree(lb):
    """
    Make all elements in this tree None, signalling not to do any more layout.
    """
    if lb is not None:
        if lb.parent is None:
            # Clear the solver.  Hopefully this garbage collects.
            lb.solver.reset()
            nonechildren(lb)
        else:
            nonetree(lb.parent)


def nonechildren(lb):
    if lb is None:
        return
    for child in lb.children.flat:
        nonechildren(child)
    lb = None


def plot_children(fig, lg, level=0, printit=False):
    """Simple plotting to show where boxes are."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig.canvas.draw()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    col = colors[level]
    for i in range(lg.nrows):
        for j in range(lg.ncols):
            bb = lg.get_outer_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=1,
                                   edgecolor='0.7', facecolor='0.7',
                                   alpha=0.2, transform=fig.transFigure,
                                   zorder=-3))
            bbi = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=2,
                                   edgecolor=col, facecolor='none',
                                   transform=fig.transFigure, zorder=-2))

            bbi = lg.get_left_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.7, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_right_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.5, 0.5],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_bottom_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.5, 0.5, 0.7],
                                   transform=fig.transFigure, zorder=-2))
            bbi = lg.get_top_margin_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0,
                                   edgecolor='none', alpha=0.2,
                                   facecolor=[0.7, 0.2, 0.7],
                                   transform=fig.transFigure, zorder=-2))
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level+1)
