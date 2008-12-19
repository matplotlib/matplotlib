import matplotlib.cbook as cbook

import matplotlib.pyplot as plt

from axes_divider import Size, SubplotDivider, LocatableAxes, Divider, get_demo_image

class AxesGrid(object):

    def __init__(self, fig, rect,
                 nrows_ncols,
                 ngrids = None,
                 direction="row",
                 axes_pad = 0.02,
                 axes_class=None,
                 add_all=True,
                 share_all=False,
                 aspect=True,
                 label_mode="L",
                 colorbar_mode=None,
                 colorbar_location="right",
                 colorbar_pad=None,
                 colorbar_size="5%",
                 ):

        self._nrows, self._ncols = nrows_ncols

        if ngrids is None:
            ngrids = self._nrows * self._ncols
        else:
            if (ngrids > self._nrows * self._ncols) or  (ngrids <= 0):
                raise Exception("")

        self.ngrids = ngrids

        self._axes_pad = axes_pad

        self._colorbar_mode = colorbar_mode
        self._colorbar_location = colorbar_location
        if colorbar_pad is None:
            self._colorbar_pad = axes_pad
        else:
            self._colorbar_pad = colorbar_pad

        self._colorbar_size = colorbar_size

        if direction not in ["column", "row"]:
            raise Exception("")

        self._direction = direction


        if axes_class is None:
            axes_class = LocatableAxes


        self.axes_all = []
        self.axes_column = [[] for i in range(self._ncols)]
        self.axes_row = [[] for i in range(self._nrows)]

        self.cbar_axes = []

        h = []
        v = []
        if cbook.is_string_like(rect) or cbook.is_numlike(rect):
            self._divider = SubplotDivider(fig, rect, horizontal=h, vertical=v,
                                           aspect=aspect)
        elif len(rect) == 3:
            kw = dict(horizontal=h, vertical=v, aspect=aspect)
            self._divider = SubplotDivider(fig, *rect, **kw)
        elif len(rect) == 4:
            self._divider = Divider(fig, rect, horizontal=h, vertical=v,
                                    aspect=aspect)
        else:
            raise Exception("")


        rect = self._divider.get_position()

        # reference axes
        self._column_refax = [None for i in range(self._ncols)]
        self._row_refax = [None for i in range(self._nrows)]
        self._refax = None

        for i in range(self.ngrids):

            col, row = self.get_col_row(i)

            if share_all:
                sharex = self._refax
                sharey = self._refax
            else:
                sharex = self._column_refax[col]
                sharey = self._row_refax[row]

            ax = axes_class(fig, rect, sharex=sharex, sharey=sharey)

            if share_all:
                if self._refax is None:
                    self._refax = ax
            else:
                if sharex is None:
                    self._column_refax[col] = ax
                if sharey is None:
                    self._row_refax[row] = ax

            self.axes_all.append(ax)
            self.axes_column[col].append(ax)
            self.axes_row[row].append(ax)

            cax = axes_class(fig, rect)
            self.cbar_axes.append(cax)

        self.axes_llc = self.axes_column[0][-1]

        self._update_locators()

        if add_all:
            for ax in self.axes_all+self.cbar_axes:
                fig.add_axes(ax)

        self.set_label_mode(label_mode)


    def _update_locators(self):

        h = []

        h_ax_pos = []
        h_cb_pos = []
        for ax in self._column_refax:
            if h: h.append(Size.Fixed(self._axes_pad))

            h_ax_pos.append(len(h))

            if ax:
                sz = Size.AxesX(ax)
            else:
                sz = Size.AxesX(self.axes_llc)
            h.append(sz)

            if self._colorbar_mode == "each" and self._colorbar_location == "right":
                h.append(Size.from_any(self._colorbar_pad, sz))
                h_cb_pos.append(len(h))
                h.append(Size.from_any(self._colorbar_size, sz))


        v = []

        v_ax_pos = []
        v_cb_pos = []
        for ax in self._row_refax[::-1]:
            if v: v.append(Size.Fixed(self._axes_pad))
            v_ax_pos.append(len(v))
            if ax:
                sz = Size.AxesY(ax)
            else:
                sz = Size.AxesY(self.axes_llc)
            v.append(sz)


            if self._colorbar_mode == "each" and self._colorbar_location == "top":
                v.append(Size.from_any(self._colorbar_pad, sz))
                v_cb_pos.append(len(v))
                v.append(Size.from_any(self._colorbar_size, sz))


        for i in range(self.ngrids):
            col, row = self.get_col_row(i)
            #locator = self._divider.new_locator(nx=4*col, ny=2*(self._nrows - row - 1))
            locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                ny=v_ax_pos[self._nrows -1 - row])
            self.axes_all[i].set_axes_locator(locator)

            if self._colorbar_mode == "each":
                if self._colorbar_location == "right":
                    locator = self._divider.new_locator(nx=h_cb_pos[col],
                                                        ny=v_ax_pos[self._nrows -1 - row])
                elif self._colorbar_location == "top":
                    locator = self._divider.new_locator(nx=h_ax_pos[col],
                                                        ny=v_cb_pos[self._nrows -1 - row])
                self.cbar_axes[i].set_axes_locator(locator)


        if self._colorbar_mode == "single":
            if self._colorbar_location == "right":
                sz = Size.Fraction(Size.AxesX(self.axes_llc), self._nrows)
                h.append(Size.from_any(self._colorbar_pad, sz))
                h.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=-2, ny=0, ny1=-1)
            elif self._colorbar_location == "top":
                sz = Size.Fraction(Size.AxesY(self.axes_llc), self._ncols)
                v.append(Size.from_any(self._colorbar_pad, sz))
                v.append(Size.from_any(self._colorbar_size, sz))
                locator = self._divider.new_locator(nx=0, nx1=-1, ny=-2)
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
            self.cbar_axes[0].set_axes_locator(locator)
            self.cbar_axes[0].set_visible(True)
        elif self._colorbar_mode == "each":
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(True)
        else:
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)

        self._divider.set_horizontal(h)
        self._divider.set_vertical(v)



    def get_col_row(self, n):
        if self._direction == "column":
            col, row = divmod(n, self._nrows)
        else:
            row, col = divmod(n, self._ncols)

        return col, row


    def __getitem__(self, i):
        return self.axes_all[i]


    def get_geometry(self):
        return self._nrows, self._ncols

    def set_axes_pad(self, axes_pad):
        self._axes_pad = axes_pad

    def get_axes_pad(self):
        return self._axes_pad

    def set_aspect(self, aspect):
        self._divider.set_aspect(aspect)

    def get_aspect(self):
        return self._divider.get_aspect()

    def set_label_mode(self, mode):
        if mode == "all":
            for ax in self.axes_all:
                [l.set_visible(True) for l in ax.get_xticklabels()]
                [l.set_visible(True) for l in ax.get_yticklabels()]
        elif mode == "L":
            for ax in self.axes_column[0][:-1]:
                [l.set_visible(False) for l in ax.get_xticklabels()]
                [l.set_visible(True) for l in ax.get_yticklabels()]
            ax = self.axes_column[0][-1]
            [l.set_visible(True) for l in ax.get_xticklabels()]
            [l.set_visible(True) for l in ax.get_yticklabels()]
            for col in self.axes_column[1:]:
                for ax in col[:-1]:
                    [l.set_visible(False) for l in ax.get_xticklabels()]
                    [l.set_visible(False) for l in ax.get_yticklabels()]
                ax = col[-1]
                [l.set_visible(True) for l in ax.get_xticklabels()]
                [l.set_visible(False) for l in ax.get_yticklabels()]
        elif mode == "1":
            for ax in self.axes_all:
                [l.set_visible(False) for l in ax.get_xticklabels()]
                [l.set_visible(False) for l in ax.get_yticklabels()]
            ax = self.axes_llc
            [l.set_visible(True) for l in ax.get_xticklabels()]
            [l.set_visible(True) for l in ax.get_yticklabels()]



if __name__ == "__main__":
    F = plt.figure(1, (9, 3.5))
    F.clf()

    F.subplots_adjust(left=0.05, right=0.98)

    grid = AxesGrid(F, 131, # similar to subplot(111)
                    nrows_ncols = (2, 2),
                    direction="row",
                    axes_pad = 0.05,
                    add_all=True,
                    label_mode = "1",
                    )

    Z, extent = get_demo_image()
    plt.ioff()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")

    # This only affects axes in first column and second row as share_all = False.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])
    plt.ion()


    grid = AxesGrid(F, 132, # similar to subplot(111)
                    nrows_ncols = (2, 2),
                    direction="row",
                    axes_pad = 0.0,
                    add_all=True,
                    share_all=True,
                    label_mode = "1",
                    colorbar_mode="single",
                    )

    Z, extent = get_demo_image()
    plt.ioff()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
    plt.colorbar(im, cax = grid.cbar_axes[0])
    plt.setp(grid.cbar_axes[0].get_yticklabels(), visible=False)

    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

    plt.ion()



    grid = AxesGrid(F, 133, # similar to subplot(122)
                    nrows_ncols = (2, 2),
                    direction="row",
                    axes_pad = 0.1,
                    add_all=True,
                    label_mode = "1",
                    share_all = True,
                    colorbar_location="top",
                    colorbar_mode="each",
                    colorbar_size="7%",
                    colorbar_pad="2%",
                    )
    plt.ioff()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
        plt.colorbar(im, cax = grid.cbar_axes[i],
                     orientation="horizontal")
        grid.cbar_axes[i].xaxis.set_ticks_position("top")
        plt.setp(grid.cbar_axes[i].get_xticklabels(), visible=False)

    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

    plt.ion()
    plt.draw()
