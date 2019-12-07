"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>

Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""

from collections import defaultdict
from functools import reduce
import math

import numpy as np

from matplotlib import artist
import matplotlib.axes as maxes
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.docstring as docstring
import matplotlib.scale as mscale
from matplotlib.axes import Axes, rcParams
from matplotlib.transforms import Bbox
from matplotlib.tri.triangulation import Triangulation

from . import art3d
from . import proj3d
from . import axis3d


@cbook.deprecated("3.2", alternative="Bbox.unit()")
def unit_bbox():
    box = Bbox(np.array([[0, 0], [1, 1]]))
    return box


@cbook._define_aliases({
    "xlim3d": ["xlim"], "ylim3d": ["ylim"], "zlim3d": ["zlim"]})
class Axes3D(Axes):
    """
    3D axes object.
    """
    name = '3d'
    _shared_z_axes = cbook.Grouper()

    def __init__(
            self, fig, rect=None, *args,
            azim=-60, elev=30, sharez=None, proj_type='persp',
            **kwargs):
        """
        Parameters
        ----------
        fig : Figure
            The parent figure.
        rect : (float, float, float, float)
            The ``(left, bottom, width, height)`` axes position.
        azim : float, optional, default: -60
            Azimuthal viewing angle.
        elev : float, optional, default: 30
            Elevation viewing angle.
        sharez : Axes3D, optional
            Other axes to share z-limits with.
        proj_type : {'persp', 'ortho'}
            The projection type, default 'persp'.
        **kwargs
            Other optional keyword arguments:

            %(Axes3D)s

        Notes
        -----
        .. versionadded:: 1.2.1
            The *sharez* parameter.
        """

        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]

        self.initial_azim = azim
        self.initial_elev = elev
        self.set_proj_type(proj_type)

        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        self.xy_dataLim = Bbox.unit()
        self.zz_dataLim = Bbox.unit()
        # inhibit autoscale_view until the axes are defined
        # they can't be defined until Axes.__init__ has been called
        self.view_init(self.initial_elev, self.initial_azim)

        self._sharez = sharez
        if sharez is not None:
            self._shared_z_axes.join(self, sharez)
            self._adjustable = 'datalim'

        super().__init__(fig, rect, frameon=True, *args, **kwargs)
        # Disable drawing of axes by base class
        super().set_axis_off()
        # Enable drawing of axes by Axes3D class
        self.set_axis_on()
        self.M = None

        # func used to format z -- fall back on major formatters
        self.fmt_zdata = None

        if self.zaxis is not None:
            self._zcid = self.zaxis.callbacks.connect(
                'units finalize', lambda: self._on_units_changed(scalez=True))
        else:
            self._zcid = None

        self.mouse_init()
        self.figure.canvas.mpl_connect(
            'motion_notify_event', self._on_move),
        self.figure.canvas.mpl_connect(
            'button_press_event', self._button_press),
        self.figure.canvas.mpl_connect(
            'button_release_event', self._button_release),
        self.set_top_view()

        self.patch.set_linewidth(0)
        # Calculate the pseudo-data width and height
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        self._pseudo_w, self._pseudo_h = pseudo_bbox[1] - pseudo_bbox[0]

        self.figure.add_axes(self)

        # mplot3d currently manages its own spines and needs these turned off
        # for bounding box calculations
        for k in self.spines.keys():
            self.spines[k].set_visible(False)

    def set_axis_off(self):
        self._axis3don = False
        self.stale = True

    def set_axis_on(self):
        self._axis3don = True
        self.stale = True

    def convert_zunits(self, z):
        """
        For artists in an axes, if the zaxis has units support,
        convert *z* using zaxis unit type

        .. versionadded:: 1.2.1

        """
        return self.zaxis.convert_units(z)

    def _process_unit_info(self, xdata=None, ydata=None, zdata=None,
                           kwargs=None):
        """
        Look for unit *kwargs* and update the axis instances as necessary

        """
        super()._process_unit_info(xdata=xdata, ydata=ydata, kwargs=kwargs)

        if self.xaxis is None or self.yaxis is None or self.zaxis is None:
            return

        if zdata is not None:
            # we only need to update if there is nothing set yet.
            if not self.zaxis.have_units():
                self.zaxis.update_units(xdata)

        # process kwargs 2nd since these will override default units
        if kwargs is not None:
            zunits = kwargs.pop('zunits', self.zaxis.units)
            if zunits != self.zaxis.units:
                self.zaxis.set_units(zunits)
                # If the units being set imply a different converter,
                # we need to update.
                if zdata is not None:
                    self.zaxis.update_units(zdata)

    def set_top_view(self):
        # this happens to be the right view for the viewing coordinates
        # moved up and to the left slightly to fit labels and axes
        xdwl = 0.95 / self.dist
        xdw = 0.9 / self.dist
        ydwl = 0.95 / self.dist
        ydw = 0.9 / self.dist
        # This is purposely using the 2D Axes's set_xlim and set_ylim,
        # because we are trying to place our viewing pane.
        super().set_xlim(-xdwl, xdw, auto=None)
        super().set_ylim(-ydwl, ydw, auto=None)

    def _init_axis(self):
        """Init 3D axes; overrides creation of regular X/Y axes."""
        self.xaxis = axis3d.XAxis('x', self.xy_viewLim.intervalx,
                                  self.xy_dataLim.intervalx, self)
        self.yaxis = axis3d.YAxis('y', self.xy_viewLim.intervaly,
                                  self.xy_dataLim.intervaly, self)
        self.zaxis = axis3d.ZAxis('z', self.zz_viewLim.intervalx,
                                  self.zz_dataLim.intervalx, self)
        for ax in self.xaxis, self.yaxis, self.zaxis:
            ax.init3d()

    def get_zaxis(self):
        """Return the ``ZAxis`` (`~.axis3d.Axis`) instance."""
        return self.zaxis

    @cbook.deprecated("3.1", alternative="xaxis", pending=True)
    @property
    def w_xaxis(self):
        return self.xaxis

    @cbook.deprecated("3.1", alternative="yaxis", pending=True)
    @property
    def w_yaxis(self):
        return self.yaxis

    @cbook.deprecated("3.1", alternative="zaxis", pending=True)
    @property
    def w_zaxis(self):
        return self.zaxis

    def _get_axis_list(self):
        return super()._get_axis_list() + (self.zaxis, )

    def unit_cube(self, vals=None):
        minx, maxx, miny, maxy, minz, maxz = vals or self.get_w_lims()
        return [(minx, miny, minz),
                (maxx, miny, minz),
                (maxx, maxy, minz),
                (minx, maxy, minz),
                (minx, miny, maxz),
                (maxx, miny, maxz),
                (maxx, maxy, maxz),
                (minx, maxy, maxz)]

    def tunit_cube(self, vals=None, M=None):
        if M is None:
            M = self.M
        xyzs = self.unit_cube(vals)
        tcube = proj3d.proj_points(xyzs, M)
        return tcube

    def tunit_edges(self, vals=None, M=None):
        tc = self.tunit_cube(vals, M)
        edges = [(tc[0], tc[1]),
                 (tc[1], tc[2]),
                 (tc[2], tc[3]),
                 (tc[3], tc[0]),

                 (tc[0], tc[4]),
                 (tc[1], tc[5]),
                 (tc[2], tc[6]),
                 (tc[3], tc[7]),

                 (tc[4], tc[5]),
                 (tc[5], tc[6]),
                 (tc[6], tc[7]),
                 (tc[7], tc[4])]
        return edges

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)

        # in the superclass, we would go through and actually deal with axis
        # scales and box/datalim. Those are all irrelevant - all we need to do
        # is make sure our coordinate system is square.
        figW, figH = self.get_figure().get_size_inches()
        fig_aspect = figH / figW
        box_aspect = 1
        pb = position.frozen()
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        self.set_position(pb1.anchored(self.get_anchor(), pb), 'active')

    @artist.allow_rasterization
    def draw(self, renderer):
        # draw the background patch
        self.patch.draw(renderer)
        self._frameon = False

        # first, set the aspect
        # this is duplicated from `axes._base._AxesBase.draw`
        # but must be called before any of the artist are drawn as
        # it adjusts the view limits and the size of the bounding box
        # of the axes
        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        # add the projection matrix to the renderer
        self.M = self.get_proj()
        renderer.M = self.M
        renderer.vvec = self.vvec
        renderer.eye = self.eye
        renderer.get_axis_position = self.get_axis_position

        # Calculate projection of collections and patches and zorder them.
        # Make sure they are drawn above the grids.
        zorder_offset = max(axis.get_zorder()
                            for axis in self._get_axis_list()) + 1
        for i, col in enumerate(
                sorted(self.collections,
                       key=lambda col: col.do_3d_projection(renderer),
                       reverse=True)):
            col.zorder = zorder_offset + i
        for i, patch in enumerate(
                sorted(self.patches,
                       key=lambda patch: patch.do_3d_projection(renderer),
                       reverse=True)):
            patch.zorder = zorder_offset + i

        if self._axis3don:
            # Draw panes first
            for axis in self._get_axis_list():
                axis.draw_pane(renderer)
            # Then axes
            for axis in self._get_axis_list():
                axis.draw(renderer)

        # Then rest
        super().draw(renderer)

    def get_axis_position(self):
        vals = self.get_w_lims()
        tc = self.tunit_cube(vals, self.M)
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return xhigh, yhigh, zhigh

    def _on_units_changed(self, scalex=False, scaley=False, scalez=False):
        """
        Callback for processing changes to axis units.

        Currently forces updates of data limits and view limits.
        """
        self.relim()
        self.autoscale_view(scalex=scalex, scaley=scaley, scalez=scalez)

    def update_datalim(self, xys, **kwargs):
        pass

    def get_autoscale_on(self):
        """
        Get whether autoscaling is applied for all axes on plot commands

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.
        """
        return super().get_autoscale_on() and self.get_autoscalez_on()

    def get_autoscalez_on(self):
        """
        Get whether autoscaling for the z-axis is applied on plot commands

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.
        """
        return self._autoscaleZon

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied on plot commands

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.

        Parameters
        ----------
        b : bool
        """
        super().set_autoscale_on(b)
        self.set_autoscalez_on(b)

    def set_autoscalez_on(self, b):
        """
        Set whether autoscaling for the z-axis is applied on plot commands

        .. versionadded:: 1.1.0

        Parameters
        ----------
        b : bool
        """
        self._autoscaleZon = b

    def set_zmargin(self, m):
        """
        Set padding of Z data limits prior to autoscaling.

        *m* times the data interval will be added to each
        end of that interval before it is used in autoscaling.

        accepts: float in range 0 to 1

        .. versionadded:: 1.1.0
        """
        if m < 0 or m > 1:
            raise ValueError("margin must be in range 0 to 1")
        self._zmargin = m
        self.stale = True

    def margins(self, *margins, x=None, y=None, z=None, tight=True):
        """
        Convenience method to set or retrieve autoscaling margins.

        Call signatures::

            margins()

        returns xmargin, ymargin, zmargin

        ::

            margins(margin)

            margins(xmargin, ymargin, zmargin)

            margins(x=xmargin, y=ymargin, z=zmargin)

            margins(..., tight=False)

        All forms above set the xmargin, ymargin and zmargin
        parameters. All keyword parameters are optional.  A single
        positional argument specifies xmargin, ymargin and zmargin.
        Passing both positional and keyword arguments for xmargin,
        ymargin, and/or zmargin is invalid.

        The *tight* parameter
        is passed to :meth:`autoscale_view`, which is executed after
        a margin is changed; the default here is *True*, on the
        assumption that when margins are specified, no additional
        padding to match tick marks is usually desired.  Setting
        *tight* to *None* will preserve the previous setting.

        Specifying any margin changes only the autoscaling; for example,
        if *xmargin* is not None, then *xmargin* times the X data
        interval will be added to each end of that interval before
        it is used in autoscaling.

        .. versionadded:: 1.1.0
        """
        if margins and x is not None and y is not None and z is not None:
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments for x, y, and/or z.')
        elif len(margins) == 1:
            x = y = z = margins[0]
        elif len(margins) == 3:
            x, y, z = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all '
                            'margins, or one for each margin (x, y, z).')

        if x is None and y is None and z is None:
            if tight is not True:
                cbook._warn_external(f'ignoring tight={tight!r} in get mode')
            return self._xmargin, self._ymargin, self._zmargin

        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)
        if z is not None:
            self.set_zmargin(z)

        self.autoscale_view(
            tight=tight, scalex=(x is not None), scaley=(y is not None),
            scalez=(z is not None)
        )

    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Convenience method for simple axis view autoscaling.
        See :meth:`matplotlib.axes.Axes.autoscale` for full explanation.
        Note that this function behaves the same, but for all
        three axes.  Therefore, 'z' can be passed for *axis*,
        and 'both' applies to all three axes.

        .. versionadded:: 1.1.0
        """
        if enable is None:
            scalex = True
            scaley = True
            scalez = True
        else:
            if axis in ['x', 'both']:
                self._autoscaleXon = scalex = bool(enable)
            else:
                scalex = False
            if axis in ['y', 'both']:
                self._autoscaleYon = scaley = bool(enable)
            else:
                scaley = False
            if axis in ['z', 'both']:
                self._autoscaleZon = scalez = bool(enable)
            else:
                scalez = False
        self.autoscale_view(tight=tight, scalex=scalex, scaley=scaley,
                            scalez=scalez)

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        # This updates the bounding boxes as to keep a record as to what the
        # minimum sized rectangular volume holds the data.
        X = np.reshape(X, -1)
        Y = np.reshape(Y, -1)
        self.xy_dataLim.update_from_data_xy(
            np.column_stack([X, Y]), not had_data)
        if Z is not None:
            Z = np.reshape(Z, -1)
            self.zz_dataLim.update_from_data_xy(
                np.column_stack([Z, Z]), not had_data)
        # Let autoscale_view figure out how to use this data.
        self.autoscale_view()

    def autoscale_view(self, tight=None, scalex=True, scaley=True,
                       scalez=True):
        """
        Autoscale the view limits using the data limits.
        See :meth:`matplotlib.axes.Axes.autoscale_view` for documentation.
        Note that this function applies to the 3D axes, and as such
        adds the *scalez* to the function arguments.

        .. versionchanged:: 1.1.0
            Function signature was changed to better match the 2D version.
            *tight* is now explicitly a kwarg and placed first.

        .. versionchanged:: 1.2.1
            This is now fully functional.
        """
        # This method looks at the rectangular volume (see above)
        # of data and decides how to scale the view portal to fit it.
        if tight is None:
            # if image data only just use the datalim
            _tight = self._tight or (
                len(self.images) > 0
                and len(self.lines) == len(self.patches) == 0)
        else:
            _tight = self._tight = bool(tight)

        if scalex and self._autoscaleXon:
            self._shared_x_axes.clean()
            x0, x1 = self.xy_dataLim.intervalx
            xlocator = self.xaxis.get_major_locator()
            x0, x1 = xlocator.nonsingular(x0, x1)
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            if not _tight:
                x0, x1 = xlocator.view_limits(x0, x1)
            self.set_xbound(x0, x1)

        if scaley and self._autoscaleYon:
            self._shared_y_axes.clean()
            y0, y1 = self.xy_dataLim.intervaly
            ylocator = self.yaxis.get_major_locator()
            y0, y1 = ylocator.nonsingular(y0, y1)
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            if not _tight:
                y0, y1 = ylocator.view_limits(y0, y1)
            self.set_ybound(y0, y1)

        if scalez and self._autoscaleZon:
            self._shared_z_axes.clean()
            z0, z1 = self.zz_dataLim.intervalx
            zlocator = self.zaxis.get_major_locator()
            z0, z1 = zlocator.nonsingular(z0, z1)
            if self._zmargin > 0:
                delta = (z1 - z0) * self._zmargin
                z0 -= delta
                z1 += delta
            if not _tight:
                z0, z1 = zlocator.view_limits(z0, z1)
            self.set_zbound(z0, z1)

    def get_w_lims(self):
        """Get 3D world limits."""
        minx, maxx = self.get_xlim3d()
        miny, maxy = self.get_ylim3d()
        minz, maxz = self.get_zlim3d()
        return minx, maxx, miny, maxy, minz, maxz

    def set_xlim3d(self, left=None, right=None, emit=True, auto=False,
                   *, xmin=None, xmax=None):
        """
        Set 3D x limits.

        See :meth:`matplotlib.axes.Axes.set_xlim` for full documentation.
        """
        if right is None and np.iterable(left):
            left, right = left
        if xmin is not None:
            if left is not None:
                raise TypeError('Cannot pass both `xmin` and `left`')
            left = xmin
        if xmax is not None:
            if right is not None:
                raise TypeError('Cannot pass both `xmax` and `right`')
            right = xmax

        self._process_unit_info(xdata=(left, right))
        left = self._validate_converted_limits(left, self.convert_xunits)
        right = self._validate_converted_limits(right, self.convert_xunits)

        old_left, old_right = self.get_xlim()
        if left is None:
            left = old_left
        if right is None:
            right = old_right

        if left == right:
            cbook._warn_external(
                f"Attempting to set identical left == right == {left} results "
                f"in singular transformations; automatically expanding.")
        reverse = left > right
        left, right = self.xaxis.get_major_locator().nonsingular(left, right)
        left, right = self.xaxis.limit_range_for_scale(left, right)
        # cast to bool to avoid bad interaction between python 3.8 and np.bool_
        left, right = sorted([left, right], reverse=bool(reverse))
        self.xy_viewLim.intervalx = (left, right)

        if auto is not None:
            self._autoscaleXon = bool(auto)

        if emit:
            self.callbacks.process('xlim_changed', self)
            # Call all of the other x-axes that are shared with this one
            for other in self._shared_x_axes.get_siblings(self):
                if other is not self:
                    other.set_xlim(self.xy_viewLim.intervalx,
                                   emit=False, auto=auto)
                    if other.figure != self.figure:
                        other.figure.canvas.draw_idle()
        self.stale = True
        return left, right

    def set_ylim3d(self, bottom=None, top=None, emit=True, auto=False,
                   *, ymin=None, ymax=None):
        """
        Set 3D y limits.

        See :meth:`matplotlib.axes.Axes.set_ylim` for full documentation.
        """
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        if ymin is not None:
            if bottom is not None:
                raise TypeError('Cannot pass both `ymin` and `bottom`')
            bottom = ymin
        if ymax is not None:
            if top is not None:
                raise TypeError('Cannot pass both `ymax` and `top`')
            top = ymax

        self._process_unit_info(ydata=(bottom, top))
        bottom = self._validate_converted_limits(bottom, self.convert_yunits)
        top = self._validate_converted_limits(top, self.convert_yunits)

        old_bottom, old_top = self.get_ylim()
        if bottom is None:
            bottom = old_bottom
        if top is None:
            top = old_top

        if bottom == top:
            cbook._warn_external(
                f"Attempting to set identical bottom == top == {bottom} "
                f"results in singular transformations; automatically "
                f"expanding.")
        swapped = bottom > top
        bottom, top = self.yaxis.get_major_locator().nonsingular(bottom, top)
        bottom, top = self.yaxis.limit_range_for_scale(bottom, top)
        if swapped:
            bottom, top = top, bottom
        self.xy_viewLim.intervaly = (bottom, top)

        if auto is not None:
            self._autoscaleYon = bool(auto)

        if emit:
            self.callbacks.process('ylim_changed', self)
            # Call all of the other y-axes that are shared with this one
            for other in self._shared_y_axes.get_siblings(self):
                if other is not self:
                    other.set_ylim(self.xy_viewLim.intervaly,
                                   emit=False, auto=auto)
                    if other.figure != self.figure:
                        other.figure.canvas.draw_idle()
        self.stale = True
        return bottom, top

    def set_zlim3d(self, bottom=None, top=None, emit=True, auto=False,
                   *, zmin=None, zmax=None):
        """
        Set 3D z limits.

        See :meth:`matplotlib.axes.Axes.set_ylim` for full documentation
        """
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        if zmin is not None:
            if bottom is not None:
                raise TypeError('Cannot pass both `zmin` and `bottom`')
            bottom = zmin
        if zmax is not None:
            if top is not None:
                raise TypeError('Cannot pass both `zmax` and `top`')
            top = zmax

        self._process_unit_info(zdata=(bottom, top))
        bottom = self._validate_converted_limits(bottom, self.convert_zunits)
        top = self._validate_converted_limits(top, self.convert_zunits)

        old_bottom, old_top = self.get_zlim()
        if bottom is None:
            bottom = old_bottom
        if top is None:
            top = old_top

        if bottom == top:
            cbook._warn_external(
                f"Attempting to set identical bottom == top == {bottom} "
                f"results in singular transformations; automatically "
                f"expanding.")
        swapped = bottom > top
        bottom, top = self.zaxis.get_major_locator().nonsingular(bottom, top)
        bottom, top = self.zaxis.limit_range_for_scale(bottom, top)
        if swapped:
            bottom, top = top, bottom
        self.zz_viewLim.intervalx = (bottom, top)

        if auto is not None:
            self._autoscaleZon = bool(auto)

        if emit:
            self.callbacks.process('zlim_changed', self)
            # Call all of the other y-axes that are shared with this one
            for other in self._shared_z_axes.get_siblings(self):
                if other is not self:
                    other.set_zlim(self.zz_viewLim.intervalx,
                                   emit=False, auto=auto)
                    if other.figure != self.figure:
                        other.figure.canvas.draw_idle()
        self.stale = True
        return bottom, top

    def get_xlim3d(self):
        return tuple(self.xy_viewLim.intervalx)
    get_xlim3d.__doc__ = maxes.Axes.get_xlim.__doc__
    if get_xlim3d.__doc__ is not None:
        get_xlim3d.__doc__ += """
        .. versionchanged:: 1.1.0
            This function now correctly refers to the 3D x-limits
        """

    def get_ylim3d(self):
        return tuple(self.xy_viewLim.intervaly)
    get_ylim3d.__doc__ = maxes.Axes.get_ylim.__doc__
    if get_ylim3d.__doc__ is not None:
        get_ylim3d.__doc__ += """
        .. versionchanged:: 1.1.0
            This function now correctly refers to the 3D y-limits.
        """

    def get_zlim3d(self):
        """Get 3D z limits."""
        return tuple(self.zz_viewLim.intervalx)

    def get_zscale(self):
        """
        Return the zaxis scale string %s

        """ % (", ".join(mscale.get_scale_names()))
        return self.zaxis.get_scale()

    # We need to slightly redefine these to pass scalez=False
    # to their calls of autoscale_view.

    def set_xscale(self, value, **kwargs):
        self.xaxis._set_scale(value, **kwargs)
        self.autoscale_view(scaley=False, scalez=False)
        self._update_transScale()
        self.stale = True

    def set_yscale(self, value, **kwargs):
        self.yaxis._set_scale(value, **kwargs)
        self.autoscale_view(scalex=False, scalez=False)
        self._update_transScale()
        self.stale = True

    def set_zscale(self, value, **kwargs):
        self.zaxis._set_scale(value, **kwargs)
        self.autoscale_view(scalex=False, scaley=False)
        self._update_transScale()
        self.stale = True

    set_xscale.__doc__, set_yscale.__doc__, set_zscale.__doc__ = map(
        """
        Set the {}-axis scale.

        Parameters
        ----------
        value : {{"linear"}}
            The axis scale type to apply.  3D axes currently only support
            linear scales; other scales yield nonsensical results.

        **kwargs
            Keyword arguments are nominally forwarded to the scale class, but
            none of them is applicable for linear scales.
        """.format,
        ["x", "y", "z"])

    def set_zticks(self, *args, **kwargs):
        """
        Set z-axis tick locations.
        See :meth:`matplotlib.axes.Axes.set_yticks` for more details.

        .. note::
            Minor ticks are not supported.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.set_ticks(*args, **kwargs)

    @cbook._make_keyword_only("3.2", "minor")
    def get_zticks(self, minor=False):
        """
        Return the z ticks as a list of locations
        See :meth:`matplotlib.axes.Axes.get_yticks` for more details.

        .. note::
            Minor ticks are not supported.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.get_ticklocs(minor=minor)

    def get_zmajorticklabels(self):
        """
        Get the ztick labels as a list of Text instances

        .. versionadded:: 1.1.0
        """
        return self.zaxis.get_majorticklabels()

    def get_zminorticklabels(self):
        """
        Get the ztick labels as a list of Text instances

        .. note::
            Minor ticks are not supported. This function was added
            only for completeness.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.get_minorticklabels()

    def set_zticklabels(self, *args, **kwargs):
        """
        Set z-axis tick labels.
        See :meth:`matplotlib.axes.Axes.set_yticklabels` for more details.

        .. note::
            Minor ticks are not supported by Axes3D objects.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.set_ticklabels(*args, **kwargs)

    def get_zticklabels(self, minor=False):
        """
        Get ztick labels as a list of Text instances.
        See :meth:`matplotlib.axes.Axes.get_yticklabels` for more details.

        .. note::
            Minor ticks are not supported.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.get_ticklabels(minor=minor)

    def zaxis_date(self, tz=None):
        """
        Sets up z-axis ticks and labels that treat the z data as dates.

        .. note::
            This function is merely provided for completeness.
            Axes3D objects do not officially support dates for ticks,
            and so this may or may not work as expected.

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.

        Parameters
        ----------
        tz : `datetime.tzinfo`, default: :rc:`timezone`
        """
        self.zaxis.axis_date(tz)

    def get_zticklines(self):
        """
        Get ztick lines as a list of Line2D instances.
        Note that this function is provided merely for completeness.
        These lines are re-calculated as the display changes.

        .. versionadded:: 1.1.0
        """
        return self.zaxis.get_ticklines()

    def clabel(self, *args, **kwargs):
        """
        This function is currently not implemented for 3D axes.
        Returns *None*.
        """
        return None

    def view_init(self, elev=None, azim=None):
        """
        Set the elevation and azimuth of the axes in degrees (not radians).

        This can be used to rotate the axes programmatically.

        'elev' stores the elevation angle in the z plane (in degrees).
        'azim' stores the azimuth angle in the (x, y) plane (in degrees).

        if elev or azim are None (default), then the initial value
        is used which was specified in the :class:`Axes3D` constructor.
        """

        self.dist = 10

        if elev is None:
            self.elev = self.initial_elev
        else:
            self.elev = elev

        if azim is None:
            self.azim = self.initial_azim
        else:
            self.azim = azim

    def set_proj_type(self, proj_type):
        """
        Set the projection type.

        Parameters
        ----------
        proj_type : {'persp', 'ortho'}
        """
        self._projection = cbook._check_getitem({
            'persp': proj3d.persp_transformation,
            'ortho': proj3d.ortho_transformation,
        }, proj_type=proj_type)

    def get_proj(self):
        """
        Create the projection matrix from the current viewing position.

        elev stores the elevation angle in the z plane
        azim stores the azimuth angle in the (x, y) plane

        dist is the distance of the eye viewing point from the object point.
        """
        # chosen for similarity with the initial view before gh-8896
        pb_aspect = np.array([4, 4, 3]) / 3.5

        relev, razim = np.pi * self.elev/180, np.pi * self.azim/180

        xmin, xmax = self.get_xlim3d()
        ymin, ymax = self.get_ylim3d()
        zmin, zmax = self.get_zlim3d()

        # transform to uniform world coordinates 0-1, 0-1, 0-1
        worldM = proj3d.world_transformation(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax, pb_aspect=pb_aspect)

        # look into the middle of the new coordinates
        R = pb_aspect / 2

        xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
        yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
        zp = R[2] + np.sin(relev) * self.dist
        E = np.array((xp, yp, zp))

        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / np.linalg.norm(self.vvec)

        if abs(relev) > np.pi/2:
            # upside down
            V = np.array((0, 0, -1))
        else:
            V = np.array((0, 0, 1))
        zfront, zback = -self.dist, self.dist

        viewM = proj3d.view_transformation(E, R, V)
        projM = self._projection(zfront, zback)
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M

    def mouse_init(self, rotate_btn=1, zoom_btn=3):
        """
        Set the mouse buttons for 3D rotation and zooming.

        Parameters
        ----------
        rotate_btn : int or list of int, default: 1
            The mouse button or buttons to use for 3D rotation of the axes.
        zoom_btn : int or list of int, default: 3
            The mouse button or buttons to use to zoom the 3D axes.
        """
        self.button_pressed = None
        # coerce scalars into array-like, then convert into
        # a regular list to avoid comparisons against None
        # which breaks in recent versions of numpy.
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()

    def disable_mouse_rotation(self):
        """Disable mouse buttons for 3D rotation and zooming."""
        self.mouse_init(rotate_btn=[], zoom_btn=[])

    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.

        3D axes objects do not use the zoom box button.
        """
        return False

    def can_pan(self):
        """
        Return *True* if this axes supports the pan/zoom button functionality.

        3D axes objects do not use the pan/zoom button.
        """
        return False

    def cla(self):
        # docstring inherited.

        super().cla()
        self.zaxis.cla()

        if self._sharez is not None:
            self.zaxis.major = self._sharez.zaxis.major
            self.zaxis.minor = self._sharez.zaxis.minor
            z0, z1 = self._sharez.get_zlim()
            self.set_zlim(z0, z1, emit=False, auto=None)
            self.zaxis._set_scale(self._sharez.zaxis.get_scale())
        else:
            self.zaxis._set_scale('linear')
            try:
                self.set_zlim(0, 1)
            except TypeError:
                pass

        self._autoscaleZon = True
        self._zmargin = 0

        self.grid(rcParams['axes3d.grid'])

    def _button_press(self, event):
        if event.inaxes == self:
            self.button_pressed = event.button
            self.sx, self.sy = event.xdata, event.ydata
            toolbar = getattr(self.figure.canvas, "toolbar")
            if toolbar and toolbar._nav_stack() is None:
                self.figure.canvas.toolbar.push_current()

    def _button_release(self, event):
        self.button_pressed = None
        toolbar = getattr(self.figure.canvas, "toolbar")
        if toolbar:
            self.figure.canvas.toolbar.push_current()

    def _get_view(self):
        # docstring inherited
        return (self.get_xlim(), self.get_ylim(), self.get_zlim(),
                self.elev, self.azim)

    def _set_view(self, view):
        # docstring inherited
        xlim, ylim, zlim, elev, azim = view
        self.set(xlim=xlim, ylim=ylim, zlim=zlim)
        self.elev = elev
        self.azim = azim

    def format_zdata(self, z):
        """
        Return *z* string formatted.  This function will use the
        :attr:`fmt_zdata` attribute if it is callable, else will fall
        back on the zaxis major formatter
        """
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            func = self.zaxis.get_major_formatter().format_data_short
            val = func(z)
            return val

    def format_coord(self, xd, yd):
        """
        Given the 2D view coordinates attempt to guess a 3D coordinate.
        Looks for the nearest edge to the point and then assumes that
        the point is at the same z location as the nearest point on the edge.
        """

        if self.M is None:
            return ''

        if self.button_pressed in self._rotate_btn:
            return 'azimuth={:.0f} deg, elevation={:.0f} deg '.format(
                self.azim, self.elev)
            # ignore xd and yd and display angles instead

        # nearest edge
        p0, p1 = min(self.tunit_edges(),
                     key=lambda edge: proj3d._line2d_seg_dist(
                         edge[0], edge[1], (xd, yd)))

        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = np.hypot(x0-xd, y0-yd)
        d1 = np.hypot(x1-xd, y1-yd)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1

        x, y, z = proj3d.inv_transform(xd, yd, z, self.M)

        xs = self.format_xdata(x)
        ys = self.format_ydata(y)
        zs = self.format_zdata(z)
        return 'x=%s, y=%s, z=%s' % (xs, ys, zs)

    def _on_move(self, event):
        """Mouse moving

        button-1 rotates by default.  Can be set explicitly in mouse_init().
        button-3 zooms by default.  Can be set explicitly in mouse_init().
        """

        if not self.button_pressed:
            return

        if self.M is None:
            return

        x, y = event.xdata, event.ydata
        # In case the mouse is out of bounds.
        if x is None:
            return

        dx, dy = x - self.sx, y - self.sy
        w = self._pseudo_w
        h = self._pseudo_h
        self.sx, self.sy = x, y

        # Rotation
        if self.button_pressed in self._rotate_btn:
            # rotate viewing point
            # get the x and y pixel coords
            if dx == 0 and dy == 0:
                return
            self.elev = art3d._norm_angle(self.elev - (dy/h)*180)
            self.azim = art3d._norm_angle(self.azim - (dx/w)*180)
            self.get_proj()
            self.stale = True
            self.figure.canvas.draw_idle()

#        elif self.button_pressed == 2:
            # pan view
            # project xv, yv, zv -> xw, yw, zw
            # pan
#            pass

        # Zoom
        elif self.button_pressed in self._zoom_btn:
            # zoom view
            # hmmm..this needs some help from clipping....
            minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
            df = 1-((h - dy)/h)
            dx = (maxx-minx)*df
            dy = (maxy-miny)*df
            dz = (maxz-minz)*df
            self.set_xlim3d(minx - dx, maxx + dx)
            self.set_ylim3d(miny - dy, maxy + dy)
            self.set_zlim3d(minz - dz, maxz + dz)
            self.get_proj()
            self.figure.canvas.draw_idle()

    def set_zlabel(self, zlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set zlabel.  See doc for `.set_ylabel` for description.
        """
        if labelpad is not None:
            self.zaxis.labelpad = labelpad
        return self.zaxis.set_label_text(zlabel, fontdict, **kwargs)

    def get_zlabel(self):
        """
        Get the z-label text string.

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.
        """
        label = self.zaxis.get_label()
        return label.get_text()

    # Axes rectangle characteristics

    def get_frame_on(self):
        """Get whether the 3D axes panels are drawn."""
        return self._frameon

    def set_frame_on(self, b):
        """
        Set whether the 3D axes panels are drawn.

        Parameters
        ----------
        b : bool
        """
        self._frameon = bool(b)
        self.stale = True

    def grid(self, b=True, **kwargs):
        """
        Set / unset 3D grid.

        .. note::

            Currently, this function does not behave the same as
            :meth:`matplotlib.axes.Axes.grid`, but it is intended to
            eventually support that behavior.

        .. versionadded:: 1.1.0
        """
        # TODO: Operate on each axes separately
        if len(kwargs):
            b = True
        self._draw_grid = b
        self.stale = True

    def locator_params(self, axis='both', tight=None, **kwargs):
        """
        Convenience method for controlling tick locators.

        See :meth:`matplotlib.axes.Axes.locator_params` for full
        documentation.  Note that this is for Axes3D objects,
        therefore, setting *axis* to 'both' will result in the
        parameters being set for all three axes.  Also, *axis*
        can also take a value of 'z' to apply parameters to the
        z axis.

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.
        """
        _x = axis in ['x', 'both']
        _y = axis in ['y', 'both']
        _z = axis in ['z', 'both']
        if _x:
            self.xaxis.get_major_locator().set_params(**kwargs)
        if _y:
            self.yaxis.get_major_locator().set_params(**kwargs)
        if _z:
            self.zaxis.get_major_locator().set_params(**kwargs)
        self.autoscale_view(tight=tight, scalex=_x, scaley=_y, scalez=_z)

    def tick_params(self, axis='both', **kwargs):
        """
        Convenience method for changing the appearance of ticks and
        tick labels.

        See :meth:`matplotlib.axes.Axes.tick_params` for more complete
        documentation.

        The only difference is that setting *axis* to 'both' will
        mean that the settings are applied to all three axes. Also,
        the *axis* parameter also accepts a value of 'z', which
        would mean to apply to only the z-axis.

        Also, because of how Axes3D objects are drawn very differently
        from regular 2D axes, some of these settings may have
        ambiguous meaning.  For simplicity, the 'z' axis will
        accept settings as if it was like the 'y' axis.

        .. note::
           Axes3D currently ignores some of these settings.

        .. versionadded:: 1.1.0
        """
        cbook._check_in_list(['x', 'y', 'z', 'both'], axis=axis)
        if axis in ['x', 'y', 'both']:
            super().tick_params(axis, **kwargs)
        if axis in ['z', 'both']:
            zkw = dict(kwargs)
            zkw.pop('top', None)
            zkw.pop('bottom', None)
            zkw.pop('labeltop', None)
            zkw.pop('labelbottom', None)
            self.zaxis.set_tick_params(**zkw)

    # data limits, ticks, tick labels, and formatting

    def invert_zaxis(self):
        """
        Invert the z-axis.

        .. versionadded:: 1.1.0
            This function was added, but not tested. Please report any bugs.
        """
        bottom, top = self.get_zlim()
        self.set_zlim(top, bottom, auto=None)

    def zaxis_inverted(self):
        """
        Returns True if the z-axis is inverted.

        .. versionadded:: 1.1.0
        """
        bottom, top = self.get_zlim()
        return top < bottom

    def get_zbound(self):
        """
        Return the lower and upper z-axis bounds, in increasing order.

        .. versionadded:: 1.1.0
        """
        bottom, top = self.get_zlim()
        if bottom < top:
            return bottom, top
        else:
            return top, bottom

    def set_zbound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the z-axis.

        This method will honor axes inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalez_on()`).

        .. versionadded:: 1.1.0
        """
        if upper is None and np.iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_zbound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        self.set_zlim(sorted((lower, upper),
                             reverse=bool(self.zaxis_inverted())),
                      auto=None)

    def text(self, x, y, z, s, zdir=None, **kwargs):
        """
        Add text to the plot. kwargs will be passed on to Axes.text,
        except for the *zdir* keyword, which sets the direction to be
        used as the z direction.
        """
        text = super().text(x, y, s, **kwargs)
        art3d.text_2d_to_3d(text, z, zdir)
        return text

    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, zdir='z', **kwargs):
        """
        Plot 2D or 3D data.

        Parameters
        ----------
        xs : 1D array-like
            x coordinates of vertices.
        ys : 1D array-like
            y coordinates of vertices.
        zs : scalar or 1D array-like
            z coordinates of vertices; either one for all points or one for
            each point.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.plot`.
        """
        had_data = self.has_data()

        # `zs` can be passed positionally or as keyword; checking whether
        # args[0] is a string matches the behavior of 2D `plot` (via
        # `_process_plot_var_args`).
        if args and not isinstance(args[0], str):
            zs, *args = args
            if 'zs' in kwargs:
                raise TypeError("plot() for multiple values for argument 'z'")
        else:
            zs = kwargs.pop('zs', 0)

        # Match length
        zs = np.broadcast_to(zs, np.shape(xs))

        lines = super().plot(xs, ys, *args, **kwargs)
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir)

        xs, ys, zs = art3d.juggle_axes(xs, ys, zs, zdir)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines

    plot3D = plot

    def plot_surface(self, X, Y, Z, *args, norm=None, vmin=None,
                     vmax=None, lightsource=None, **kwargs):
        """
        Create a surface plot.

        By default it will be colored in shades of a solid color, but it also
        supports color mapping by supplying the *cmap* argument.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        Parameters
        ----------
        X, Y, Z : 2d arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Defaults to 50.

            .. versionadded:: 2.0

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 10.

            'classic' mode uses a default of ``rstride = cstride = 10`` instead
            of the new default of ``rcount = ccount = 50``.

        color : color-like
            Color of the surface patches.

        cmap : Colormap
            Colormap of the surface patches.

        facecolors : array-like of colors.
            Colors of each individual patch.

        norm : Normalize
            Normalization for the colormap.

        vmin, vmax : float
            Bounds for the normalization.

        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        **kwargs
            Other arguments are forwarded to `.Poly3DCollection`.
        """

        had_data = self.has_data()

        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        if np.any(np.isnan(Z)):
            cbook._warn_external(
                "Z contains NaN values. This may result in rendering "
                "artifacts.")

        # TODO: Support masked arrays
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 10)
        cstride = kwargs.pop('cstride', 10)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            compute_strides = has_count
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            compute_strides = not has_stride

        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))

        if 'facecolors' in kwargs:
            fcolors = kwargs.pop('facecolors')
        else:
            color = kwargs.pop('color', None)
            if color is None:
                color = self._get_lines.get_next_color()
            color = np.array(mcolors.to_rgba(color))
            fcolors = None

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        if shade is None:
            cbook.warn_deprecated(
                "3.1",
                message="Passing shade=None to Axes3D.plot_surface() is "
                        "deprecated since matplotlib 3.1 and will change its "
                        "semantic or raise an error in matplotlib 3.3. "
                        "Please use shade=False instead.")

        # evenly spaced, and including both endpoints
        row_inds = list(range(0, rows-1, rstride)) + [rows-1]
        col_inds = list(range(0, cols-1, cstride)) + [cols-1]

        colset = []  # the sampled facecolor
        polys = []
        for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
            for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                ps = [
                    # +1 ensures we share edges between polygons
                    cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
                    for a in (X, Y, Z)
                ]
                # ps = np.stack(ps, axis=-1)
                ps = np.array(ps).T
                polys.append(ps)

                if fcolors is not None:
                    colset.append(fcolors[rs][cs])

        # note that the striding causes some polygons to have more coordinates
        # than others
        polyc = art3d.Poly3DCollection(polys, *args, **kwargs)

        if fcolors is not None:
            if shade:
                colset = self._shade_colors(
                    colset, self._generate_normals(polys), lightsource)
            polyc.set_facecolors(colset)
            polyc.set_edgecolors(colset)
        elif cmap:
            # doesn't vectorize because polys is jagged
            avg_z = np.array([ps[:, 2].mean() for ps in polys])
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            if shade:
                colset = self._shade_colors(
                    color, self._generate_normals(polys), lightsource)
            else:
                colset = color
            polyc.set_facecolors(colset)

        self.add_collection(polyc)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return polyc

    def _generate_normals(self, polygons):
        """
        Takes a list of polygons and return an array of their normals.

        Normals point towards the viewer for a face with its vertices in
        counterclockwise order, following the right hand rule.

        Uses three points equally spaced around the polygon.
        This normal of course might not make sense for polygons with more than
        three points not lying in a plane, but it's a plausible and fast
        approximation.

        Parameters
        ----------
        polygons: list of (M_i, 3) array-like, or (..., M, 3) array-like
            A sequence of polygons to compute normals for, which can have
            varying numbers of vertices. If the polygons all have the same
            number of vertices and array is passed, then the operation will
            be vectorized.

        Returns
        -------
        normals: (..., 3) array-like
            A normal vector estimated for the polygon.

        """
        if isinstance(polygons, np.ndarray):
            # optimization: polygons all have the same number of points, so can
            # vectorize
            n = polygons.shape[-2]
            i1, i2, i3 = 0, n//3, 2*n//3
            v1 = polygons[..., i1, :] - polygons[..., i2, :]
            v2 = polygons[..., i2, :] - polygons[..., i3, :]
        else:
            # The subtraction doesn't vectorize because polygons is jagged.
            v1 = np.empty((len(polygons), 3))
            v2 = np.empty((len(polygons), 3))
            for poly_i, ps in enumerate(polygons):
                n = len(ps)
                i1, i2, i3 = 0, n//3, 2*n//3
                v1[poly_i, :] = ps[i1, :] - ps[i2, :]
                v2[poly_i, :] = ps[i2, :] - ps[i3, :]
        return np.cross(v1, v2)

    def _shade_colors(self, color, normals, lightsource=None):
        """
        Shade *color* using normal vectors given by *normals*.
        *color* can also be an array of the same length as *normals*.
        """
        if lightsource is None:
            # chosen for backwards-compatibility
            lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)

        with np.errstate(invalid="ignore"):
            shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
                     @ lightsource.direction)
        mask = ~np.isnan(shade)

        if mask.any():
            # convert dot product to allowed shading fractions
            in_norm = mcolors.Normalize(-1, 1)
            out_norm = mcolors.Normalize(0.3, 1).inverse

            def norm(x):
                return out_norm(in_norm(x))

            shade[~mask] = 0

            color = mcolors.to_rgba_array(color)
            # shape of color should be (M, 4) (where M is number of faces)
            # shape of shade should be (M,)
            # colors should have final shape of (M, 4)
            alpha = color[:, 3]
            colors = norm(shade)[:, np.newaxis] * color
            colors[:, 3] = alpha
        else:
            colors = np.asanyarray(color).copy()

        return colors

    def plot_wireframe(self, X, Y, Z, *args, **kwargs):
        """
        Plot a 3D wireframe.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        Parameters
        ----------
        X, Y, Z : 2d arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Setting a count to zero causes the data to be
            not sampled in the corresponding direction, producing a 3D line
            plot rather than a wireframe plot.  Defaults to 50.

            .. versionadded:: 2.0

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 1.  Setting a
            stride to zero causes the data to be not sampled in the
            corresponding direction, producing a 3D line plot rather than a
            wireframe plot.

            'classic' mode uses a default of ``rstride = cstride = 1`` instead
            of the new default of ``rcount = ccount = 50``.

        **kwargs
            Other arguments are forwarded to `.Line3DCollection`.
        """

        had_data = self.has_data()
        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        # FIXME: Support masked arrays
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 1)
        cstride = kwargs.pop('cstride', 1)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            if has_count:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            if not has_stride:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0

        # We want two sets of lines, one running along the "rows" of
        # Z and another set of lines running along the "columns" of Z.
        # This transpose will make it easy to obtain the columns.
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        if rstride:
            rii = list(range(0, rows, rstride))
            # Add the last index only if needed
            if rows > 0 and rii[-1] != (rows - 1):
                rii += [rows-1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            # Add the last index only if needed
            if cols > 0 and cii[-1] != (cols - 1):
                cii += [cols-1]
        else:
            cii = []

        if rstride == 0 and cstride == 0:
            raise ValueError("Either rstride or cstride must be non zero")

        # If the inputs were empty, then just
        # reset everything.
        if Z.size == 0:
            rii = []
            cii = []

        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = ([list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(xlines, ylines, zlines)]
                 + [list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(txlines, tylines, tzlines)])

        linec = art3d.Line3DCollection(lines, *args, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return linec

    def plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None,
                     lightsource=None, **kwargs):
        """
        Plot a triangulated surface.

        The (optional) triangulation can be specified in one of two ways;
        either::

          plot_trisurf(triangulation, ...)

        where triangulation is a :class:`~matplotlib.tri.Triangulation`
        object, or::

          plot_trisurf(X, Y, ...)
          plot_trisurf(X, Y, triangles, ...)
          plot_trisurf(X, Y, triangles=triangles, ...)

        in which case a Triangulation object will be created.  See
        :class:`~matplotlib.tri.Triangulation` for a explanation of
        these possibilities.

        The remaining arguments are::

          plot_trisurf(..., Z)

        where *Z* is the array of values to contour, one per point
        in the triangulation.

        Parameters
        ----------
        X, Y, Z : array-like
            Data values as 1D arrays.
        color
            Color of the surface patches.
        cmap
            A colormap for the surface patches.
        norm : Normalize
            An instance of Normalize to map values to colors.
        vmin, vmax : scalar, optional, default: None
            Minimum and maximum value to map.
        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.
        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.
        **kwargs
            All other arguments are passed on to
            :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

        Examples
        --------
        .. plot:: gallery/mplot3d/trisurf3d.py
        .. plot:: gallery/mplot3d/trisurf3d_2.py

        .. versionadded:: 1.2.0
        """

        had_data = self.has_data()

        # TODO: Support custom face colours
        if color is None:
            color = self._get_lines.get_next_color()
        color = np.array(mcolors.to_rgba(color))

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)

        tri, args, kwargs = \
            Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        try:
            z = kwargs.pop('Z')
        except KeyError:
            # We do this so Z doesn't get passed as an arg to PolyCollection
            z, *args = args
        z = np.asarray(z)

        triangles = tri.get_masked_triangles()
        xt = tri.x[triangles]
        yt = tri.y[triangles]
        zt = z[triangles]
        verts = np.stack((xt, yt, zt), axis=-1)

        polyc = art3d.Poly3DCollection(verts, *args, **kwargs)

        if cmap:
            # average over the three points of each triangle
            avg_z = verts[:, :, 2].mean(axis=1)
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            if shade:
                normals = self._generate_normals(verts)
                colset = self._shade_colors(color, normals, lightsource)
            else:
                colset = color
            polyc.set_facecolors(colset)

        self.add_collection(polyc)
        self.auto_scale_xyz(tri.x, tri.y, z, had_data)

        return polyc

    def _3d_extend_contour(self, cset, stride=5):
        """
        Extend a contour in 3D by creating
        """

        levels = cset.levels
        colls = cset.collections
        dz = (levels[1] - levels[0]) / 2

        for z, linec in zip(levels, colls):
            paths = linec.get_paths()
            if not paths:
                continue
            topverts = art3d._paths_to_3d_segments(paths, z - dz)
            botverts = art3d._paths_to_3d_segments(paths, z + dz)

            color = linec.get_color()[0]

            polyverts = []
            normals = []
            nsteps = round(len(topverts[0]) / stride)
            if nsteps <= 1:
                if len(topverts[0]) > 1:
                    nsteps = 2
                else:
                    continue

            stepsize = (len(topverts[0]) - 1) / (nsteps - 1)
            for i in range(int(round(nsteps)) - 1):
                i1 = int(round(i * stepsize))
                i2 = int(round((i + 1) * stepsize))
                polyverts.append([topverts[0][i1],
                                  topverts[0][i2],
                                  botverts[0][i2],
                                  botverts[0][i1]])

            # all polygons have 4 vertices, so vectorize
            polyverts = np.array(polyverts)
            normals = self._generate_normals(polyverts)

            colors = self._shade_colors(color, normals)
            colors2 = self._shade_colors(color, normals)
            polycol = art3d.Poly3DCollection(polyverts,
                                             facecolors=colors,
                                             edgecolors=colors2)
            polycol.set_sort_zpos(z)
            self.add_collection3d(polycol)

        for col in colls:
            self.collections.remove(col)

    def add_contour_set(
            self, cset, extend3d=False, stride=5, zdir='z', offset=None):
        zdir = '-' + zdir
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            for z, linec in zip(cset.levels, cset.collections):
                if offset is not None:
                    z = offset
                art3d.line_collection_2d_to_3d(linec, z, zdir=zdir)

    def add_contourf_set(self, cset, zdir='z', offset=None):
        zdir = '-' + zdir
        for z, linec in zip(cset.levels, cset.collections):
            if offset is not None:
                z = offset
            art3d.poly_collection_2d_to_3d(linec, z, zdir=zdir)
            linec.set_sort_zpos(z)

    def contour(self, X, Y, Z, *args,
                extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        Parameters
        ----------
        X, Y, Z : array-likes
            Input data.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : scalar
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contour`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contour(jX, jY, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contour3D = contour

    def tricontour(self, *args,
                   extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        .. versionchanged:: 1.3.0
            Added support for custom triangulations

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-likes
            Input data.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : scalar
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.

        Returns
        -------
        matplotlib.tri.tricontour.TriContourSet
        """
        had_data = self.has_data()

        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            # We do this so Z doesn't get passed as an arg to Axes.tricontour
            Z, *args = args

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)

        cset = super().tricontour(tri, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    def contourf(self, X, Y, Z, *args, zdir='z', offset=None, **kwargs):
        """
        Create a 3D filled contour plot.

        Parameters
        ----------
        X, Y, Z : array-likes
            Input data.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : scalar
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contourf`.

        Returns
        -------
        matplotlib.contour.QuadContourSet

        Notes
        -----
        .. versionadded:: 1.1.0
            The *zdir* and *offset* parameters.
        """
        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contourf(jX, jY, jZ, *args, **kwargs)
        self.add_contourf_set(cset, zdir, offset)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contourf3D = contourf

    def tricontourf(self, *args, zdir='z', offset=None, **kwargs):
        """
        Create a 3D filled contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-likes
            Input data.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : scalar
            If specified, plot a projection of the contour lines at this
            position in a plane normal to zdir
        *args, **kwargs
            Other arguments are forwarded to
            `matplotlib.axes.Axes.tricontourf`.

        Returns
        -------
        matplotlib.tri.tricontour.TriContourSet

        Notes
        -----
        .. versionadded:: 1.1.0
            The *zdir* and *offset* parameters.
        .. versionchanged:: 1.3.0
            Added support for custom triangulations
        """
        had_data = self.has_data()

        tri, args, kwargs = Triangulation.get_from_args_and_kwargs(
                *args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            # We do this so Z doesn't get passed as an arg to Axes.tricontourf
            Z, *args = args

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)

        cset = super().tricontourf(tri, jZ, *args, **kwargs)
        self.add_contourf_set(cset, zdir, offset)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    def add_collection3d(self, col, zs=0, zdir='z'):
        """
        Add a 3D collection object to the plot.

        2D collection types are converted to a 3D version by
        modifying the object and adding z coordinate information.

        Supported are:
            - PolyCollection
            - LineCollection
            - PatchCollection
        """
        zvals = np.atleast_1d(zs)
        zsortval = (np.min(zvals) if zvals.size
                    else 0)  # FIXME: arbitrary default

        # FIXME: use issubclass() (although, then a 3D collection
        #       object would also pass.)  Maybe have a collection3d
        #       abstract class to test for and exclude?
        if type(col) is mcoll.PolyCollection:
            art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.LineCollection:
            art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.PatchCollection:
            art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)

        super().add_collection(col)

    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,
                *args, **kwargs):
        """
        Create a scatter plot.

        Parameters
        ----------
        xs, ys : array-like
             The data positions.
        zs : float or array-like, optional, default: 0
            The z-positions. Either an array of the same length as *xs* and
            *ys* or a single value to place all points in the same plane.
        zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, optional, default: 'z'
            The axis direction for the *zs*. This is useful when plotting 2D
            data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting
            *zdir* to 'y' then plots the data to the x-z-plane.

            See also :doc:`/gallery/mplot3d/2dcollections3d`.

        s : scalar or array-like, optional, default: 20
            The marker size in points**2. Either an array of the same length
            as *xs* and *ys* or a single value to make all markers the same
            size.
        c : color, sequence, or sequence of colors, optional
            The marker color. Possible values:

            - A single color format string.
            - A sequence of colors of length n.
            - A sequence of n numbers to be mapped to colors using *cmap* and
              *norm*.
            - A 2-D array in which the rows are RGB or RGBA.

            For more details see the *c* argument of `~.axes.Axes.scatter`.
        depthshade : bool, optional, default: True
            Whether to shade the scatter markers to give the appearance of
            depth. Each call to ``scatter()`` will perform its depthshading
            independently.
        **kwargs
            All other arguments are passed on to `~.axes.Axes.scatter`.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`
        """

        had_data = self.has_data()
        zs_orig = zs

        xs, ys, zs = np.broadcast_arrays(
            *[np.ravel(np.ma.filled(t, np.nan)) for t in [xs, ys, zs]])
        s = np.ma.ravel(s)  # This doesn't have to match x, y in size.

        xs, ys, zs, s, c = cbook.delete_masked_points(xs, ys, zs, s, c)

        # For xs and ys, 2D scatter() will do the copying.
        if np.may_share_memory(zs_orig, zs):  # Avoid unnecessary copies.
            zs = zs.copy()

        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir,
                                        depthshade=depthshade)

        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)

        self.auto_scale_xyz(xs, ys, zs, had_data)

        return patches

    scatter3D = scatter

    def bar(self, left, height, zs=0, zdir='z', *args, **kwargs):
        """
        Add 2D bar(s).

        Parameters
        ----------
        left : 1D array-like
            The x coordinates of the left sides of the bars.
        height : 1D array-like
            The height of the bars.
        zs : scalar or 1D array-like
            Z coordinate of bars; if a single value is specified, it will be
            used for all bars.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Patch3DCollection
        """
        had_data = self.has_data()

        patches = super().bar(left, height, *args, **kwargs)

        zs = np.broadcast_to(zs, len(left))

        verts = []
        verts_zs = []
        for p, z in zip(patches, zs):
            vs = art3d._get_patch_verts(p)
            verts += vs.tolist()
            verts_zs += [z] * len(vs)
            art3d.patch_2d_to_3d(p, z, zdir)
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])

        if len(verts) > 0:
            # the following has to be skipped if verts is empty
            # NOTE: Bugs could still occur if len(verts) > 0,
            #       but the "2nd dimension" is empty.
            xs, ys = zip(*verts)
        else:
            xs, ys = [], []

        xs, ys, verts_zs = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)

        return patches

    def bar3d(self, x, y, z, dx, dy, dz, color=None,
              zsort='average', shade=True, lightsource=None, *args, **kwargs):
        """Generate a 3D barplot.

        This method creates three dimensional barplot where the width,
        depth, height, and color of the bars can all be uniquely set.

        Parameters
        ----------
        x, y, z : array-like
            The coordinates of the anchor point of the bars.

        dx, dy, dz : scalar or array-like
            The width, depth, and height of the bars, respectively.

        color : sequence of colors, optional
            The color of the bars can be specified globally or
            individually. This parameter can be:

            - A single color, to color all bars the same color.
            - An array of colors of length N bars, to color each bar
              independently.
            - An array of colors of length 6, to color the faces of the
              bars similarly.
            - An array of colors of length 6 * N bars, to color each face
              independently.

            When coloring the faces of the boxes specifically, this is
            the order of the coloring:

              1. -Z (bottom of box)
              2. +Z (top of box)
              3. -Y
              4. +Y
              5. -X
              6. +X

        zsort : str, optional
            The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`

        shade : bool, optional, default: True
            When true, this shades the dark sides of the bars (relative
            to the plot's source of light).

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        **kwargs
            Any additional keyword arguments are passed onto
            `~.art3d.Poly3DCollection`.

        Returns
        -------
        collection : `~.art3d.Poly3DCollection`
            A collection of three dimensional polygons representing
            the bars.
        """

        had_data = self.has_data()

        x, y, z, dx, dy, dz = np.broadcast_arrays(
            np.atleast_1d(x), y, z, dx, dy, dz)
        minx = np.min(x)
        maxx = np.max(x + dx)
        miny = np.min(y)
        maxy = np.max(y + dy)
        minz = np.min(z)
        maxz = np.max(z + dz)

        # shape (6, 4, 3)
        # All faces are oriented facing outwards - when viewed from the
        # outside, their vertices are in a counterclockwise ordering.
        cuboid = np.array([
            # -z
            (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ),
            # +z
            (
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            # -y
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
            ),
            # +y
            (
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (1, 1, 0),
            ),
            # -x
            (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
            ),
            # +x
            (
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
            ),
        ])

        # indexed by [bar, face, vertex, coord]
        polys = np.empty(x.shape + cuboid.shape)

        # handle each coordinate separately
        for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
            p = p[..., np.newaxis, np.newaxis]
            dp = dp[..., np.newaxis, np.newaxis]
            polys[..., i] = p + dp * cuboid[..., i]

        # collapse the first two axes
        polys = polys.reshape((-1,) + polys.shape[2:])

        facecolors = []
        if color is None:
            color = [self._get_patches_for_fill.get_next_color()]

        if len(color) == len(x):
            # bar colors specified, need to expand to number of faces
            for c in color:
                facecolors.extend([c] * 6)
        else:
            # a single color specified, or face colors specified explicitly
            facecolors = list(mcolors.to_rgba_array(color))
            if len(facecolors) < len(x):
                facecolors *= (6 * len(x))

        if shade:
            normals = self._generate_normals(polys)
            sfacecolors = self._shade_colors(facecolors, normals, lightsource)
        else:
            sfacecolors = facecolors

        col = art3d.Poly3DCollection(polys,
                                     zsort=zsort,
                                     facecolor=sfacecolors,
                                     *args, **kwargs)
        self.add_collection(col)

        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

        return col

    def set_title(self, label, fontdict=None, loc='center', **kwargs):
        # docstring inherited
        ret = super().set_title(label, fontdict=fontdict, loc=loc, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)
        return ret

    def quiver(self, *args,
               length=1, arrow_length_ratio=.3, pivot='tail', normalize=False,
               **kwargs):
        """
        ax.quiver(X, Y, Z, U, V, W, /, length=1, arrow_length_ratio=.3, \
pivot='tail', normalize=False, **kwargs)

        Plot a 3D field of arrows.

        The arguments could be array-like or scalars, so long as they
        they can be broadcast together. The arguments can also be
        masked arrays. If an element in any of argument is masked, then
        that corresponding quiver element will not be plotted.

        Parameters
        ----------
        X, Y, Z : array-like
            The x, y and z coordinates of the arrow locations (default is
            tail of arrow; see *pivot* kwarg).

        U, V, W : array-like
            The x, y and z components of the arrow vectors.

        length : float, default: 1
            The length of each quiver.

        arrow_length_ratio : float, default: 0.3
            The ratio of the arrow head with respect to the quiver.

        pivot : {'tail', 'middle', 'tip'}, default: 'tail'
            The part of the arrow that is at the grid point; the arrow
            rotates about this point, hence the name *pivot*.

        normalize : bool, default: False
            Whether all arrows are normalized to have the same length, or keep
            the lengths defined by *u*, *v*, and *w*.

        **kwargs
            Any additional keyword arguments are delegated to
            :class:`~matplotlib.collections.LineCollection`
        """
        def calc_arrows(UVW, angle=15):
            # get unit direction vector perpendicular to (u, v, w)
            x = UVW[:, 0]
            y = UVW[:, 1]
            norm = np.linalg.norm(UVW[:, :2], axis=1)
            x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
            y_p = np.divide(-x,  norm, where=norm != 0, out=np.ones_like(x))

            # compute the two arrowhead direction unit vectors
            ra = math.radians(angle)
            c = math.cos(ra)
            s = math.sin(ra)

            # construct the rotation matrices
            Rpos = np.array(
                [[c + (x_p ** 2) * (1 - c), x_p * y_p * (1 - c), y_p * s],
                 [y_p * x_p * (1 - c), c + (y_p ** 2) * (1 - c), -x_p * s],
                 [-y_p * s, x_p * s, np.full_like(x_p, c)]])
            Rpos = Rpos.transpose(2, 0, 1)

            # opposite rotation negates all the sin terms
            Rneg = Rpos.copy()
            Rneg[:, [0, 1, 2, 2], [2, 2, 0, 1]] = \
                -Rneg[:, [0, 1, 2, 2], [2, 2, 0, 1]]

            # expand dimensions for batched matrix multiplication
            UVW = np.expand_dims(UVW, axis=-1)

            # multiply them to get the rotated vector
            Rpos_vecs = np.matmul(Rpos, UVW)
            Rneg_vecs = np.matmul(Rneg, UVW)

            # transpose for concatenation
            Rpos_vecs = Rpos_vecs.transpose(0, 2, 1)
            Rneg_vecs = Rneg_vecs.transpose(0, 2, 1)

            head_dirs = np.concatenate([Rpos_vecs, Rneg_vecs], axis=1)

            return head_dirs

        had_data = self.has_data()

        # handle args
        argi = 6
        if len(args) < argi:
            raise ValueError('Wrong number of arguments. Expected %d got %d' %
                             (argi, len(args)))

        # first 6 arguments are X, Y, Z, U, V, W
        input_args = args[:argi]

        # extract the masks, if any
        masks = [k.mask for k in input_args
                 if isinstance(k, np.ma.MaskedArray)]
        # broadcast to match the shape
        bcast = np.broadcast_arrays(*input_args, *masks)
        input_args = bcast[:argi]
        masks = bcast[argi:]
        if masks:
            # combine the masks into one
            mask = reduce(np.logical_or, masks)
            # put mask on and compress
            input_args = [np.ma.array(k, mask=mask).compressed()
                          for k in input_args]
        else:
            input_args = [np.ravel(k) for k in input_args]

        if any(len(v) == 0 for v in input_args):
            # No quivers, so just make an empty collection and return early
            linec = art3d.Line3DCollection([], *args[argi:], **kwargs)
            self.add_collection(linec)
            return linec

        shaft_dt = np.array([0., length], dtype=float)
        arrow_dt = shaft_dt * arrow_length_ratio

        cbook._check_in_list(['tail', 'middle', 'tip'], pivot=pivot)
        if pivot == 'tail':
            shaft_dt -= length
        elif pivot == 'middle':
            shaft_dt -= length / 2

        XYZ = np.column_stack(input_args[:3])
        UVW = np.column_stack(input_args[3:argi]).astype(float)

        # Normalize rows of UVW
        norm = np.linalg.norm(UVW, axis=1)

        # If any row of UVW is all zeros, don't make a quiver for it
        mask = norm > 0
        XYZ = XYZ[mask]
        if normalize:
            UVW = UVW[mask] / norm[mask].reshape((-1, 1))
        else:
            UVW = UVW[mask]

        if len(XYZ) > 0:
            # compute the shaft lines all at once with an outer product
            shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
            # compute head direction vectors, n heads x 2 sides x 3 dimensions
            head_dirs = calc_arrows(UVW)
            # compute all head lines at once, starting from the shaft ends
            heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
            # stack left and right head lines together
            heads.shape = (len(arrow_dt), -1, 3)
            # transpose to get a list of lines
            heads = heads.swapaxes(0, 1)

            lines = [*shafts, *heads]
        else:
            lines = []

        linec = art3d.Line3DCollection(lines, *args[argi:], **kwargs)
        self.add_collection(linec)

        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)

        return linec

    quiver3D = quiver

    def voxels(self, *args, facecolors=None, edgecolors=None, shade=True,
               lightsource=None, **kwargs):
        """
        ax.voxels([x, y, z,] /, filled, facecolors=None, edgecolors=None, \
**kwargs)

        Plot a set of filled voxels

        All voxels are plotted as 1x1x1 cubes on the axis, with
        ``filled[0, 0, 0]`` placed with its lower corner at the origin.
        Occluded faces are not plotted.

        .. versionadded:: 2.1

        Parameters
        ----------
        filled : 3D np.array of bool
            A 3d array of values, with truthy values indicating which voxels
            to fill

        x, y, z : 3D np.array, optional
            The coordinates of the corners of the voxels. This should broadcast
            to a shape one larger in every dimension than the shape of
            *filled*.  These can be used to plot non-cubic voxels.

            If not specified, defaults to increasing integers along each axis,
            like those returned by :func:`~numpy.indices`.
            As indicated by the ``/`` in the function signature, these
            arguments can only be passed positionally.

        facecolors, edgecolors : array-like, optional
            The color to draw the faces and edges of the voxels. Can only be
            passed as keyword arguments.
            This parameter can be:

              - A single color value, to color all voxels the same color. This
                can be either a string, or a 1D rgb/rgba array
              - ``None``, the default, to use a single color for the faces, and
                the style default for the edges.
              - A 3D ndarray of color names, with each item the color for the
                corresponding voxel. The size must match the voxels.
              - A 4D ndarray of rgb/rgba data, with the components along the
                last axis.

        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.

            .. versionadded:: 3.1

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

            .. versionadded:: 3.1

        **kwargs
            Additional keyword arguments to pass onto
            `~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

        Returns
        -------
        faces : dict
            A dictionary indexed by coordinate, where ``faces[i, j, k]`` is a
            `.Poly3DCollection` of the faces drawn for the voxel
            ``filled[i, j, k]``. If no faces were drawn for a given voxel,
            either because it was not asked to be drawn, or it is fully
            occluded, then ``(i, j, k) not in faces``.

        Examples
        --------
        .. plot:: gallery/mplot3d/voxels.py
        .. plot:: gallery/mplot3d/voxels_rgb.py
        .. plot:: gallery/mplot3d/voxels_torus.py
        .. plot:: gallery/mplot3d/voxels_numpy_logo.py
        """

        # work out which signature we should be using, and use it to parse
        # the arguments. Name must be voxels for the correct error message
        if len(args) >= 3:
            # underscores indicate position only
            def voxels(__x, __y, __z, filled, **kwargs):
                return (__x, __y, __z), filled, kwargs
        else:
            def voxels(filled, **kwargs):
                return None, filled, kwargs

        xyz, filled, kwargs = voxels(*args, **kwargs)

        # check dimensions
        if filled.ndim != 3:
            raise ValueError("Argument filled must be 3-dimensional")
        size = np.array(filled.shape, dtype=np.intp)

        # check xyz coordinates, which are one larger than the filled shape
        coord_shape = tuple(size + 1)
        if xyz is None:
            x, y, z = np.indices(coord_shape)
        else:
            x, y, z = (np.broadcast_to(c, coord_shape) for c in xyz)

        def _broadcast_color_arg(color, name):
            if np.ndim(color) in (0, 1):
                # single color, like "red" or [1, 0, 0]
                return np.broadcast_to(color, filled.shape + np.shape(color))
            elif np.ndim(color) in (3, 4):
                # 3D array of strings, or 4D array with last axis rgb
                if np.shape(color)[:3] != filled.shape:
                    raise ValueError(
                        "When multidimensional, {} must match the shape of "
                        "filled".format(name))
                return color
            else:
                raise ValueError("Invalid {} argument".format(name))

        # broadcast and default on facecolors
        if facecolors is None:
            facecolors = self._get_patches_for_fill.get_next_color()
        facecolors = _broadcast_color_arg(facecolors, 'facecolors')

        # broadcast but no default on edgecolors
        edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

        # scale to the full array, even if the data is only in the center
        self.auto_scale_xyz(x, y, z)

        # points lying on corners of a square
        square = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.intp)

        voxel_faces = defaultdict(list)

        def permutation_matrices(n):
            """Generator of cyclic permutation matrices."""
            mat = np.eye(n, dtype=np.intp)
            for i in range(n):
                yield mat
                mat = np.roll(mat, 1, axis=0)

        # iterate over each of the YZ, ZX, and XY orientations, finding faces
        # to render
        for permute in permutation_matrices(3):
            # find the set of ranges to iterate over
            pc, qc, rc = permute.T.dot(size)
            pinds = np.arange(pc)
            qinds = np.arange(qc)
            rinds = np.arange(rc)

            square_rot_pos = square.dot(permute.T)
            square_rot_neg = square_rot_pos[::-1]

            # iterate within the current plane
            for p in pinds:
                for q in qinds:
                    # iterate perpendicularly to the current plane, handling
                    # boundaries. We only draw faces between a voxel and an
                    # empty space, to avoid drawing internal faces.

                    # draw lower faces
                    p0 = permute.dot([p, q, 0])
                    i0 = tuple(p0)
                    if filled[i0]:
                        voxel_faces[i0].append(p0 + square_rot_neg)

                    # draw middle faces
                    for r1, r2 in zip(rinds[:-1], rinds[1:]):
                        p1 = permute.dot([p, q, r1])
                        p2 = permute.dot([p, q, r2])

                        i1 = tuple(p1)
                        i2 = tuple(p2)

                        if filled[i1] and not filled[i2]:
                            voxel_faces[i1].append(p2 + square_rot_pos)
                        elif not filled[i1] and filled[i2]:
                            voxel_faces[i2].append(p2 + square_rot_neg)

                    # draw upper faces
                    pk = permute.dot([p, q, rc-1])
                    pk2 = permute.dot([p, q, rc])
                    ik = tuple(pk)
                    if filled[ik]:
                        voxel_faces[ik].append(pk2 + square_rot_pos)

        # iterate over the faces, and generate a Poly3DCollection for each
        # voxel
        polygons = {}
        for coord, faces_inds in voxel_faces.items():
            # convert indices into 3D positions
            if xyz is None:
                faces = faces_inds
            else:
                faces = []
                for face_inds in faces_inds:
                    ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                    face = np.empty(face_inds.shape)
                    face[:, 0] = x[ind]
                    face[:, 1] = y[ind]
                    face[:, 2] = z[ind]
                    faces.append(face)

            # shade the faces
            facecolor = facecolors[coord]
            edgecolor = edgecolors[coord]
            if shade:
                normals = self._generate_normals(faces)
                facecolor = self._shade_colors(facecolor, normals, lightsource)
                if edgecolor is not None:
                    edgecolor = self._shade_colors(
                        edgecolor, normals, lightsource
                    )

            poly = art3d.Poly3DCollection(
                faces, facecolors=facecolor, edgecolors=edgecolor, **kwargs)
            self.add_collection3d(poly)
            polygons[coord] = poly

        return polygons


docstring.interpd.update(Axes3D=artist.kwdoc(Axes3D))
docstring.dedent_interpd(Axes3D.__init__)


def get_test_data(delta=0.05):
    """Return a tuple X, Y, Z with a test data set."""
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z
