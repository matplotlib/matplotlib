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
import itertools
import math
import textwrap
import warnings

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation

from . import art3d
from . import proj3d
from . import axis3d


@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):
    name = '3d'

    _axis_names = ("x", "y", "z")
    Axes._shared_axes["z"] = cbook.Grouper()
    Axes._shared_axes["view"] = cbook.Grouper()

    def __init__(
        self, fig, rect=None, *args,
        elev=30, azim=-60, roll=0, shareview=None, sharez=None,
        proj_type='persp', focal_length=None,
        box_aspect=None,
        computed_zorder=True,
        **kwargs,
    ):
        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]

        self.initial_azim = azim
        self.initial_elev = elev
        self.initial_roll = roll
        self.set_proj_type(proj_type, focal_length)
        self.computed_zorder = computed_zorder

        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        xymargin = 0.05 * 10/11
        self.xy_dataLim = Bbox([[xymargin, xymargin],
                                [1 - xymargin, 1 - xymargin]])
        self.zz_dataLim = Bbox.unit()

        self.view_init(self.initial_elev, self.initial_azim, self.initial_roll)

        self._sharez = sharez
        if sharez is not None:
            self._shared_axes["z"].join(self, sharez)
            self._adjustable = 'datalim'

        self._shareview = shareview
        if shareview is not None:
            self._shared_axes["view"].join(self, shareview)

        if kwargs.pop('auto_add_to_figure', False):
            raise AttributeError(
                'auto_add_to_figure is no longer supported for Axes3D. '
                'Use fig.add_axes(ax) instead.'
            )

        super().__init__(
            fig, rect, frameon=True, box_aspect=box_aspect, *args, **kwargs
        )
        super().set_axis_off()
        self.set_axis_on()
        self.M = None
        self.invM = None

        self._view_margin = 1/48
        self.autoscale_view()
        self.fmt_zdata = None

        self.mouse_init()
        fig = self.get_figure(root=True)
        fig.canvas.callbacks._connect_picklable('motion_notify_event', self._on_move)
        fig.canvas.callbacks._connect_picklable('button_press_event', self._button_press)
        fig.canvas.callbacks._connect_picklable('button_release_event', self._button_release)
        self.set_top_view()

        self.patch.set_linewidth(0)
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        self._pseudo_w, self._pseudo_h = pseudo_bbox[1] - pseudo_bbox[0]
        self.spines[:].set_visible(False)

        def set_axis_off(self):
        self._axis3don = False
        self.stale = True

    def set_axis_on(self):
        self._axis3don = True
        self.stale = True

    def convert_zunits(self, z):
        return self.zaxis.convert_units(z)

    def set_top_view(self):
        xdwl = 0.95 / self._dist
        xdw = 0.9 / self._dist
        ydwl = 0.95 / self._dist
        ydw = 0.9 / self._dist
        self.viewLim.intervalx = (-xdwl, xdw)
        self.viewLim.intervaly = (-ydwl, ydw)
        self.stale = True

    def _init_axis(self):
        self.xaxis = axis3d.XAxis(self)
        self.yaxis = axis3d.YAxis(self)
        self.zaxis = axis3d.ZAxis(self)

    def get_zaxis(self):
        return self.zaxis

    get_zgridlines = _axis_method_wrapper("zaxis", "get_gridlines")
    get_zticklines = _axis_method_wrapper("zaxis", "get_ticklines")

    # --- UPDATED set_zlim to support Timestamps ---
    def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, zmin=None, zmax=None):
        if zmin is not None:
            if bottom is not None:
                raise TypeError("Cannot set both 'bottom' and 'zmin'")
            bottom = zmin
        if zmax is not None:
            if top is not None:
                raise TypeError("Cannot set both 'top' and 'zmax'")
            top = zmax

        # Convert units immediately to handle Timestamps/Dates
        bottom = self.convert_zunits(bottom)
        top = self.convert_zunits(top)

        return self._set_lim3d(self.zaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=None, axmax=None,
                               minpos=self.zz_dataLim.minposx)

    set_xlim3d = Axes.set_xlim
    set_ylim3d = Axes.set_ylim
    set_zlim3d = set_zlim

    def get_xlim(self):
        return tuple(self.xy_viewLim.intervalx)

    def get_ylim(self):
        return tuple(self.xy_viewLim.intervaly)

    def get_zlim(self):
        return tuple(self.zz_viewLim.intervalx)
    
    def draw(self, renderer):
        if not self.get_visible():
            return
        self._unstale_viewLim()
        self.patch.draw(renderer)
        self._frameon = False

        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        self.M = self.get_proj()
        self.invM = np.linalg.inv(self.M)

        collections_and_patches = (
            artist for artist in self._children
            if isinstance(artist, (mcoll.Collection, mpatches.Patch))
            and artist.get_visible())

        if self.computed_zorder:
            zorder_offset = max(axis.get_zorder()
                                for axis in self._axis_map.values()) + 1
            collection_zorder = patch_zorder = zorder_offset

            for artist in sorted(collections_and_patches,
                                 key=lambda artist: artist.do_3d_projection(),
                                 reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            for artist in collections_and_patches:
                artist.do_3d_projection()

        if self._axis3don:
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            for axis in self._axis_map.values():
                axis.draw_grid(renderer)
            for axis in self._axis_map.values():
                axis.draw(renderer)

        super().draw(renderer)

    def get_proj(self):
        box_aspect = self._roll_to_vertical(self._box_aspect)
        scaled_limits = self._get_scaled_limits()
        worldM = proj3d.world_transformation(*scaled_limits, pb_aspect=box_aspect)
        R = 0.5 * box_aspect
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)
        ps = self._roll_to_vertical([p0, p1, p2])
        eye = R + self._dist * ps
        u, v, w = self._calc_view_axes(eye)
        if self._focal_length == np.inf:
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist, self._dist, self._focal_length)
        M0 = np.dot(viewM, worldM)
        return np.dot(projM, M0)

        # --- UPDATED scatter to support Timestamps ---
    @_preprocess_data(replace_names=["xs", "ys", "zs", "s",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"])
    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=None,
                *args, depthshade_minalpha=None, axlim_clip=False, **kwargs):
        
        # FIX: Convert units immediately to numeric values
        xs = self.convert_xunits(xs)
        ys = self.convert_yunits(ys)
        zs = self.convert_zunits(zs)

        # Ensure they are numpy arrays for the 3D engine math
        xs, ys, zs = np.atleast_1d(xs, ys, zs)

        had_data = self.has_data()
        zs_orig = zs

        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)
        s = np.ma.ravel(s)

        xs, ys, zs, s, c, color = cbook.delete_masked_points(
            xs, ys, zs, s, c, kwargs.get('color', None)
        )
        
        if kwargs.get("color") is not None:
            kwargs['color'] = color
        if depthshade is None:
            depthshade = mpl.rcParams['axes3d.depthshade']
        if depthshade_minalpha is None:
            depthshade_minalpha = mpl.rcParams['axes3d.depthshade_minalpha']

        if np.may_share_memory(zs_orig, zs):
            zs = zs.copy()

        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        art3d.patch_collection_2d_to_3d(
            patches, zs=zs, zdir=zdir, depthshade=depthshade,
            depthshade_minalpha=depthshade_minalpha, axlim_clip=axlim_clip,
        )
        
        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)

        self.auto_scale_xyz(xs, ys, zs, had_data)
        return patches

    def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
        self.button_pressed = None
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        self._pan_btn = np.atleast_1d(pan_btn).tolist()
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()


    def plot_surface(self, X, Y, Z, *, norm=None, vmin=None,
                     vmax=None, lightsource=None, axlim_clip=False, **kwargs):
        """Create a surface plot."""
        had_data = self.has_data()

        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")

        Z = cbook._to_unmasked_float_array(Z)
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

        compute_strides = not has_stride
        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))

        fcolors = kwargs.pop('facecolors', None)
        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)

        polys = []
        colset = []
        row_inds = list(range(0, rows-1, rstride)) + [rows-1]
        col_inds = list(range(0, cols-1, cstride)) + [cols-1]

        for rs, rs_next in itertools.pairwise(row_inds):
            for cs, cs_next in itertools.pairwise(col_inds):
                ps = [cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
                      for a in (X, Y, Z)]
                polys.append(np.array(ps).T)
                if fcolors is not None:
                    colset.append(fcolors[rs][cs])

        if fcolors is not None:
            polyc = art3d.Poly3DCollection(
                polys, edgecolors=colset, facecolors=colset, shade=shade,
                lightsource=lightsource, axlim_clip=axlim_clip, **kwargs)
        elif cmap:
            polyc = art3d.Poly3DCollection(polys, axlim_clip=axlim_clip, **kwargs)
            avg_z = np.array([ps[:, 2].mean() for ps in polys])
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            color = kwargs.pop('color', None) or self._get_lines.get_next_color()
            polyc = art3d.Poly3DCollection(
                polys, facecolors=color, shade=shade, lightsource=lightsource,
                axlim_clip=axlim_clip, **kwargs)

        self.add_collection(polyc, autolim="_datalim_only")
        self.auto_scale_xyz(X, Y, Z, had_data)
        return polyc

    def plot_wireframe(self, X, Y, Z, *, axlim_clip=False, **kwargs):
        """Plot a 3D wireframe."""
        had_data = self.has_data()
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        rstride = kwargs.pop('rstride', 1)
        cstride = kwargs.pop('cstride', 1)
        
        rii = np.arange(0, rows, rstride)
        cii = np.arange(0, cols, cstride)

        row_lines = np.stack([X[rii], Y[rii], Z[rii]], axis=-1)
        col_lines = np.stack([X.T[cii], Y.T[cii], Z.T[cii]], axis=-1)

        self.auto_scale_xyz(X, Y, Z, had_data)

        lines = list(row_lines) + list(col_lines)
        linec = art3d.Line3DCollection(lines, axlim_clip=axlim_clip, **kwargs)
        self.add_collection(linec, autolim="_datalim_only")
        return linec
    
    def contour(self, X, Y, Z, *args, zdir='z', offset=None, **kwargs):
        """Create a 3D contour plot."""
        had_data = self.has_data()
        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contour(jX, jY, jZ, *args, **kwargs)
        
        art3d.collection_2d_to_3d(
            cset, zs=offset if offset is not None else cset.levels, zdir=zdir)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    def bar3d(self, x, y, z, dx, dy, dz, color=None, zsort='average', 
              shade=True, axlim_clip=False, **kwargs):
        """Generate a 3D barplot."""
        had_data = self.has_data()
        x, y, z, dx, dy, dz = np.broadcast_arrays(np.atleast_1d(x), y, z, dx, dy, dz)

        cuboid = np.array([
            ((0,0,0), (0,1,0), (1,1,0), (1,0,0)), # -z
            ((0,0,1), (1,0,1), (1,1,1), (0,1,1)), # +z
            ((0,0,0), (1,0,0), (1,0,1), (0,0,1)), # -y
            ((0,1,0), (0,1,1), (1,1,1), (1,1,0)), # +y
            ((0,0,0), (0,0,1), (0,1,1), (0,1,0)), # -x
            ((1,0,0), (1,1,0), (1,1,1), (1,0,1)), # +x
        ])

        polys = np.empty(x.shape + cuboid.shape)
        for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
            polys[..., i] = p[..., None, None] + dp[..., None, None] * cuboid[..., i]

        polys = polys.reshape((-1,) + polys.shape[2:])
        color = color or self._get_patches_for_fill.get_next_color()
        
        col = art3d.Poly3DCollection(polys, zsort=zsort, facecolors=color, 
                                     shade=shade, axlim_clip=axlim_clip, **kwargs)
        self.add_collection(col, autolim="_datalim_only")
        self.auto_scale_xyz((x.min(), (x+dx).max()), (y.min(), (y+dy).max()), 
                           (z.min(), (z+dz).max()), had_data)
        return col
    
    def quiver(self, X, Y, Z, U, V, W, *, length=1, arrow_length_ratio=.3, **kwargs):
        """Plot a 3D field of arrows."""
        had_data = self.has_data()
        input_args = cbook._broadcast_with_masks(X, Y, Z, U, V, W, compress=True)
        XYZ = np.column_stack(input_args[:3])
        UVW = np.column_stack(input_args[3:]).astype(float)

        shaft_dt = np.array([0., length])
        shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)

        linec = art3d.Line3DCollection(shafts, **kwargs)
        self.add_collection(linec, autolim="_datalim_only")
        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)
        return linec

    def stem(self, x, y, z, *, orientation='z', **kwargs):
        """Create a 3D stem plot."""
        from matplotlib.container import StemContainer
        had_data = self.has_data()
        
        # Logic to create vertical lines from bottom to z
        lines = [[(thisx, thisy, 0), (thisx, thisy, thisz)] 
                 for thisx, thisy, thisz in zip(x, y, z)]
        
        stemlines = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(stemlines, autolim="_datalim_only")
        markerline, = self.plot(x, y, z, 'o')
        
        self.auto_scale_xyz(x, y, z, had_data)
        return StemContainer((markerline, stemlines), label=kwargs.get('label'))