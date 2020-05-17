"""

Conventions:

"constrain_x" means to constrain the variable with either
another kiwisolver variable, or a float.  i.e. `constrain_width(0.2)`
will set a constraint that the width has to be 0.2 and this constraint is
permanent - i.e. it will not be removed if it becomes obsolete.

"edit_x" means to set x to a value (just a float), and that this value can
change.  So `edit_width(0.2)` will set width to be 0.2, but `edit_width(0.3)`
will allow it to change to 0.3 later.  Note that these values are still just
"suggestions" in `kiwisolver` parlance, and could be over-ridden by
other constrains.

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
    Basic rectangle representation using kiwi solver variables
    """

    def __init__(self, parent=None, parent_pos=(0, 0),
                 parent_inner=False, name='', ncols=1, nrows=1,
                 h_pad=None, w_pad=None, width_ratios=None,
                 height_ratios=None, fixed_margins=False):
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
            # parent wants to know about this child!

        sol = self.solver

        # keep track of artist associated w/ this layout.  Can be none
        self.artists = np.empty((nrows, ncols), dtype=object)
        self.children = np.empty((nrows, ncols), dtype=object)

        self.fixed_margins = fixed_margins
        self.margins = {}
        self.margin_vals = {}
        # all the layout boxes in each row share the same top/bottom margins
        for todo in ['left', 'right']:
            self.margins[todo] = np.empty((ncols), dtype=object)
            self.margin_vals[todo] = np.zeros(ncols)

        # these are actually all slaves to the parent and the
        # margins, but its useful to define them all
        self.widths = np.empty((ncols), dtype=object)
        self.lefts = np.empty((ncols), dtype=object)
        self.rights = np.empty((ncols), dtype=object)
        self.inner_widths = np.empty((ncols), dtype=object)

        for i in range(self.ncols):
            for todo in ['left', 'right']:
                self.margins[todo][i] = Variable(f'{sn}margins[{todo}][{i}]')
                sol.addEditVariable(self.margins[todo][i], 'strong')
                # self.margins[todo][i] = Variable(f'{sn}margins[{todo}][{i}]')
            self.rights[i] = Variable(f'{sn}rights[{i}]')
            self.lefts[i] = Variable(f'{sn}lefts[{i}]')
            self.widths[i] = Variable(f'{sn}widths[{i}]')
            self.inner_widths[i] = Variable(f'{sn}inner_widths[{i}]')

        for todo in ['bottom', 'top']:
            self.margins[todo] = np.empty((nrows), dtype=object)
            # self.margins_min[todo] = np.empty((nrows), dtype=object)
            self.margin_vals[todo] = np.zeros(nrows)

        self.heights = np.empty((nrows), dtype=object)
        self.inner_heights = np.empty((nrows), dtype=object)
        self.bottoms = np.empty((nrows), dtype=object)
        self.tops = np.empty((nrows), dtype=object)

        for i in range(self.nrows):
            for todo in ['bottom', 'top']:
#                self.margins[todo][i] = Variable(f'{sn}margins[{todo}][{i}]')
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
                       f'MR{self.margins["right"][j].value():1.3f},\n ' #\
#                       f'MML{self.margins_min["left"][j].value():1.3f}, ' \
#                       f'MMR{self.margins_min["right"][j].value():1.3f}\n'
        return str

    def reset_margins(self):
        for todo in ['left', 'right', 'bottom', 'top']:
            self.edit_margins(todo, 0.0)


    def add_constraints(self):
        # define relation ships between things thing width and right and left
        self.hard_constraints()
        # self.soft_constraints()
        self.parent_constrain()
        #        self.solver.dump()
        self.grid_constraints()
        # sol.updateVariables()

    def grid_constraints(self):
        w0 = self.inner_widths[0] / self.width_ratios[0]
        # from left to right
        for i in range(1, self.ncols):
            w = self.inner_widths[i]
            c = (w == w0 * self.width_ratios[i])
            self.solver.addConstraint(c | 'required')
            c = (self.rights[i-1] == self.lefts[i])
            self.solver.addConstraint(c | 'required')
        h0 = self.inner_heights[0] / self.height_ratios[0]
        # from top to bottom:
        for i in range(1, self.nrows):
            h = self.inner_heights[i]
            c = (h == h0 * self.height_ratios[i])
            self.solver.addConstraint(c | 'required')
            c = (self.bottoms[i-1] == self.tops[i])
            self.solver.addConstraint(c | 'required')

    def hard_constraints(self):
        for i in range(self.ncols):
            hc = [self.widths[i] == self.rights[i] - self.lefts[i],
                  self.widths[i] >= 0,
                  self.inner_widths[i] == (
                          self.rights[i] - self.margins['right'][i] -
                        self.lefts[i] - self.margins['left'][i])]
            for c in hc:
                self.solver.addConstraint(c | 'required')
            for c in hc:
                self.solver.addConstraint(c | 'required')

        for i in range(self.nrows):
            hc = [self.heights[i] == self.tops[i] - self.bottoms[i],
                  self.heights[i] >= 0,
                  self.inner_heights[i] == (
                        self.tops[i] - self.margins['top'][i] -
                        self.bottoms[i] - self.margins['bottom'][i]),
                  ]
            for c in hc:
                self.solver.addConstraint(c | 'required')

    def add_child(self, child, i=0, j=0):
        self.children[i, j] = child

    def parent_constrain(self):
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

    # Margin editing:  The margins are variable and meant to
    # contain things of a fixes size.
    def edit_margin(self, todo, width, col):
        "update the margin at col by width"
        self.solver.suggestValue(self.margins[todo][col], width)
        self.margin_vals[todo][col] = width

    def edit_margin_min(self, todo, width, col=0):
        "update the margin at col by width is width is greater than margin"
        if width > self.margin_vals[todo][col]:
            self.edit_margin(todo, width, col)

    def edit_margins(self, todo, width):
        for i in range(len(self.margin_vals[todo])):
            self.edit_margin(todo, width, i)

    def edit_margins_min(self, todo, width):
        for i in range(len(self.margin_vals[todo])):
            self.edit_margin_min(todo, width, i)

    def get_margins(self, todo, col):
        "Return the margin at this position"
        return self.margin_vals[todo][col]

    def match_width(self, col1, col2):
        # right-left
        w1 = (self.rights[col1.stop-1] - self.margins['right'][col1.stop-1] -
              (self.lefts[col1.start] + self.margins['left'][col1.start]))
        w2 = (self.rights[col2.stop-1] - self.margins['right'][col2.stop-1] -
              (self.lefts[col2.start] + self.margins['left'][col2.start]))

        # TODO: fix width rations
        c = (w1 == w2)
        # self.solver.addConstraint(c | 'strong')


    def get_outer_bbox(self, rows=[0], cols=[0]):
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)

        bbox = Bbox.from_extents(
            self.lefts[cols[0]].value(),
            self.bottoms[rows[-1]].value(),
            self.rights[cols[-1]].value(),
            self.tops[rows[0]].value())
        return bbox

    def get_inner_bbox(self, rows=[0], cols=[0]):
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

    def update_variables(self):
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


def print_tree(lb):
    """Print the tree of layoutboxes."""

    if lb.parent is None:
        print('LayoutBox Tree\n')
        print('==============\n')
        print_children(lb)
        print('\n')
    else:
        print_tree(lb.parent)

def plot_children(fig, lg, level=0, printit=False):
    """Simple plotting to show where boxes are."""
    import matplotlib
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
                      edgecolor='0.7', facecolor='0.7', alpha=0.2,
                      transform=fig.transFigure, zorder=-3))

            bb = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(
                mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=2,
                          edgecolor=col, facecolor='none',
                          transform=fig.transFigure, zorder=-2))
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level+1)