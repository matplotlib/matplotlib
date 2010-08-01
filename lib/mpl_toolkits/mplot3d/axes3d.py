#!/usr/bin/python
# axes3d.py, original mplot3d version by John Porter
# Created: 23 Sep 2005
# Parts fixed by Reinier Heeres <reinier@heeres.eu>
# Minor additions by Ben Axelrod <baxelrod@coroware.com>

"""
Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""

import warnings
from matplotlib.axes import Axes, rcParams
from matplotlib import cbook
from matplotlib.transforms import Bbox
from matplotlib import collections
import numpy as np
from matplotlib.colors import Normalize, colorConverter, LightSource

import art3d
import proj3d
import axis3d

def sensible_format_data(self, value):
    """Used to generate more comprehensible numbers in status bar"""
    if abs(value) > 1e4 or abs(value)<1e-3:
        s = '%1.4e' % value
        return self._formatSciNotation(s)
    else:
        return '%4.3f' % value

def unit_bbox():
    box = Bbox(np.array([[0, 0], [1, 1]]))
    return box

class Axes3D(Axes):
    """
    3D axes object.
    """
    name = '3d'

    def __init__(self, fig, rect=None, *args, **kwargs):
        '''
        Build an :class:`Axes3D` instance in
        :class:`~matplotlib.figure.Figure` *fig* with
        *rect=[left, bottom, width, height]* in
        :class:`~matplotlib.figure.Figure` coordinates

        Optional keyword arguments:

          ================   =========================================
          Keyword            Description
          ================   =========================================
          *azim*             Azimuthal viewing angle (default -60)
          *elev*             Elevation viewing angle (default 30)
          ================   =========================================
        '''

        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]
        self.fig = fig
        self._cids = []

        self.initial_azim = kwargs.pop('azim', -60)
        self.initial_elev = kwargs.pop('elev', 30)

        self.xy_viewLim = unit_bbox()
        self.zz_viewLim = unit_bbox()
        self.xy_dataLim = unit_bbox()
        self.zz_dataLim = unit_bbox()
        # inihibit autoscale_view until the axes are defined
        # they can't be defined until Axes.__init__ has been called
        self.view_init(self.initial_elev, self.initial_azim)
        self._ready = 0
        Axes.__init__(self, self.fig, rect,
                      frameon=True,
                      xticks=[], yticks=[], *args, **kwargs)

        self.M = None

        self._ready = 1
        self.mouse_init()
        self.create_axes()
        self.set_top_view()

        self.axesPatch.set_linewidth(0)
        self.fig.add_axes(self)

    def set_top_view(self):
        # this happens to be the right view for the viewing coordinates
        # moved up and to the left slightly to fit labels and axes
        xdwl = (0.95/self.dist)
        xdw = (0.9/self.dist)
        ydwl = (0.95/self.dist)
        ydw = (0.9/self.dist)

        Axes.set_xlim(self, -xdwl, xdw, auto=None)
        Axes.set_ylim(self, -ydwl, ydw, auto=None)

    def create_axes(self):
        self.w_xaxis = axis3d.XAxis('x', self.xy_viewLim.intervalx,
                            self.xy_dataLim.intervalx, self)
        self.w_yaxis = axis3d.YAxis('y', self.xy_viewLim.intervaly,
                            self.xy_dataLim.intervaly, self)
        self.w_zaxis = axis3d.ZAxis('z', self.zz_viewLim.intervalx,
                            self.zz_dataLim.intervalx, self)

    def unit_cube(self, vals=None):
        minx, maxx, miny, maxy, minz, maxz = vals or self.get_w_lims()
        xs, ys, zs = ([minx, maxx, maxx, minx, minx, maxx, maxx, minx],
                    [miny, miny, maxy, maxy, miny, miny, maxy, maxy],
                    [minz, minz, minz, minz, maxz, maxz, maxz, maxz])
        return zip(xs, ys, zs)

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

    def draw(self, renderer):
        # draw the background patch
        self.axesPatch.draw(renderer)
        self._frameon = False

        # add the projection matrix to the renderer
        self.M = self.get_proj()
        renderer.M = self.M
        renderer.vvec = self.vvec
        renderer.eye = self.eye
        renderer.get_axis_position = self.get_axis_position

        # Calculate projection of collections and zorder them
        zlist = [(col.do_3d_projection(renderer), col) \
                 for col in self.collections]
        zlist.sort()
        zlist.reverse()
        for i, (z, col) in enumerate(zlist):
            col.zorder = i

        # Calculate projection of patches and zorder them
        zlist = [(patch.do_3d_projection(renderer), patch) \
                for patch in self.patches]
        zlist.sort()
        zlist.reverse()
        for i, (z, patch) in enumerate(zlist):
            patch.zorder = i

        axes = (self.w_xaxis, self.w_yaxis, self.w_zaxis)
        for ax in axes:
            ax.draw_pane(renderer)
        for ax in axes:
            ax.draw(renderer)

        Axes.draw(self, renderer)

    def get_axis_position(self):
        vals = self.get_w_lims()
        tc = self.tunit_cube(vals, self.M)
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return xhigh, yhigh, zhigh

    def update_datalim(self, xys, **kwargs):
        pass

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        x, y, z = map(np.asarray, (X, Y, Z))
        try:
            x, y = x.flatten(), y.flatten()
            if Z is not None:
                z = z.flatten()
        except AttributeError:
            raise

        # This updates the bounding boxes as to keep a record as
        # to what the minimum sized rectangular volume holds the
        # data.
        self.xy_dataLim.update_from_data_xy(np.array([x, y]).T, not had_data)
        if z is not None:
            self.zz_dataLim.update_from_data_xy(np.array([z, z]).T, not had_data)

        # Let autoscale_view figure out how to use this data.
        self.autoscale_view()

    def autoscale_view(self, scalex=True, scaley=True, scalez=True, **kw):
        # This method looks at the rectanglular volume (see above)
        # of data and decides how to scale the view portal to fit it.

        self.set_top_view()
        if not self._ready:
            return

        if not self.get_autoscale_on():
            return
        if scalex:
            self.set_xlim3d(self.xy_dataLim.intervalx)
        if scaley:
            self.set_ylim3d(self.xy_dataLim.intervaly)
        if scalez:
            self.set_zlim3d(self.zz_dataLim.intervalx)

    def get_w_lims(self):
        '''Get 3d world limits.'''
        minx, maxx = self.get_xlim3d()
        miny, maxy = self.get_ylim3d()
        minz, maxz = self.get_zlim3d()
        return minx, maxx, miny, maxy, minz, maxz

    def _determine_lims(self, xmin=None, xmax=None, *args, **kwargs):
        if xmax is None and cbook.iterable(xmin):
            xmin, xmax = xmin
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        return (xmin, xmax)

    def set_xlim3d(self, *args, **kwargs):
        '''Set 3D x limits.'''
        lims = self._determine_lims(*args, **kwargs)
        self.xy_viewLim.intervalx = lims
        return lims

    def set_ylim3d(self, *args, **kwargs):
        '''Set 3D y limits.'''
        lims = self._determine_lims(*args, **kwargs)
        self.xy_viewLim.intervaly = lims
        return lims

    def set_zlim3d(self, *args, **kwargs):
        '''Set 3D z limits.'''
        lims = self._determine_lims(*args, **kwargs)
        self.zz_viewLim.intervalx = lims
        return lims

    def get_xlim3d(self):
        '''Get 3D x limits.'''
        return self.xy_viewLim.intervalx

    def get_ylim3d(self):
        '''Get 3D y limits.'''
        return self.xy_viewLim.intervaly

    def get_zlim3d(self):
        '''Get 3D z limits.'''
        return self.zz_viewLim.intervalx

    def clabel(self, *args, **kwargs):
        return None

    def pany(self, numsteps):
        print 'numsteps', numsteps

    def panpy(self, numsteps):
        print 'numsteps', numsteps

    def view_init(self, elev=None, azim=None):
        """
        Set the elevation and azimuth of the axes.

        This can be used to rotate the axes programatically.

        'elev' stores the elevation angle in the z plane.
        'azim' stores the azimuth angle in the x,y plane.

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

    def get_proj(self):
        """Create the projection matrix from the current viewing
        position.

        elev stores the elevation angle in the z plane
        azim stores the azimuth angle in the x,y plane

        dist is the distance of the eye viewing point from the object
        point.

        """
        relev, razim = np.pi * self.elev/180, np.pi * self.azim/180

        xmin, xmax = self.get_xlim3d()
        ymin, ymax = self.get_ylim3d()
        zmin, zmax = self.get_zlim3d()

        # transform to uniform world coordinates 0-1.0,0-1.0,0-1.0
        worldM = proj3d.world_transformation(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax)

        # look into the middle of the new coordinates
        R = np.array([0.5, 0.5, 0.5])

        xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
        yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
        zp = R[2] + np.sin(relev) * self.dist
        E = np.array((xp, yp, zp))

        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / proj3d.mod(self.vvec)

        if abs(relev) > np.pi/2:
            # upside down
            V = np.array((0, 0, -1))
        else:
            V = np.array((0, 0, 1))
        zfront, zback = -self.dist, self.dist

        viewM = proj3d.view_transformation(E, R, V)
        perspM = proj3d.persp_transformation(zfront, zback)
        M0 = np.dot(viewM, worldM)
        M = np.dot(perspM, M0)
        return M

    def mouse_init(self, rotate_btn=1, zoom_btn=3):
        """Initializes mouse button callbacks to enable 3D rotation of
        the axes.  Also optionally sets the mouse buttons for 3D rotation
        and zooming.

        ============  =======================================================
        Argument      Description
        ============  =======================================================
        *rotate_btn*  The integer or list of integers specifying which mouse
                      button or buttons to use for 3D rotation of the axes.
                      Default = 1.

        *zoom_btn*    The integer or list of integers specifying which mouse
                      button or buttons to use to zoom the 3D axes.
                      Default = 3.
        ============  =======================================================

        """
        self.button_pressed = None
        canv = self.figure.canvas
        if canv != None:
            c1 = canv.mpl_connect('motion_notify_event', self._on_move)
            c2 = canv.mpl_connect('button_press_event', self._button_press)
            c3 = canv.mpl_connect('button_release_event', self._button_release)
            self._cids = [c1, c2, c3]
        else:
            warnings.warn('Axes3D.figure.canvas is \'None\', mouse rotation disabled.  Set canvas then call Axes3D.mouse_init().')

        self._rotate_btn = np.atleast_1d(rotate_btn)
        self._zoom_btn = np.atleast_1d(zoom_btn)

    def cla(self):
        """Clear axes and disable mouse button callbacks.
        """
        self.disable_mouse_rotation()
        Axes.cla(self)
        self.grid(rcParams['axes3d.grid'])

    def disable_mouse_rotation(self):
        """Disable mouse button callbacks.
        """
        # Disconnect the various events we set.
        for cid in self._cids:
            self.figure.canvas.mpl_disconnect(cid)

        self._cids = []

    def _button_press(self, event):
        if event.inaxes == self:
            self.button_pressed = event.button
            self.sx, self.sy = event.xdata, event.ydata

    def _button_release(self, event):
        self.button_pressed = None

    def format_xdata(self, x):
        """
        Return x string formatted.  This function will use the attribute
        self.fmt_xdata if it is callable, else will fall back on the xaxis
        major formatter
        """
        try:
            return self.fmt_xdata(x)
        except TypeError:
            fmt = self.w_xaxis.get_major_formatter()
            return sensible_format_data(fmt, x)

    def format_ydata(self, y):
        """
        Return y string formatted.  This function will use the attribute
        self.fmt_ydata if it is callable, else will fall back on the yaxis
        major formatter
        """
        try:
            return self.fmt_ydata(y)
        except TypeError:
            fmt = self.w_yaxis.get_major_formatter()
            return sensible_format_data(fmt, y)

    def format_zdata(self, z):
        """
        Return z string formatted.  This function will use the attribute
        self.fmt_zdata if it is callable, else will fall back on the yaxis
        major formatter
        """
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            fmt = self.w_zaxis.get_major_formatter()
            return sensible_format_data(fmt, z)

    def format_coord(self, xd, yd):
        """
        Given the 2D view coordinates attempt to guess a 3D coordinate.
        Looks for the nearest edge to the point and then assumes that
        the point is at the same z location as the nearest point on the edge.
        """

        if self.M is None:
            return ''

        if self.button_pressed in self._rotate_btn:
            return 'azimuth=%d deg, elevation=%d deg ' % (self.azim, self.elev)
            # ignore xd and yd and display angles instead

        p = (xd, yd)
        edges = self.tunit_edges()
        #lines = [proj3d.line2d(p0,p1) for (p0,p1) in edges]
        ldists = [(proj3d.line2d_seg_dist(p0, p1, p), i) for \
                i, (p0, p1) in enumerate(edges)]
        ldists.sort()
        # nearest edge
        edgei = ldists[0][1]

        p0, p1 = edges[edgei]

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
        zs = self.format_ydata(z)
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
        if x == None:
            return

        dx, dy = x - self.sx, y - self.sy
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        w = (x1-x0)
        h = (y1-y0)
        self.sx, self.sy = x, y

        # Rotation
        if self.button_pressed in self._rotate_btn:
            # rotate viewing point
            # get the x and y pixel coords
            if dx == 0 and dy == 0:
                return
            self.elev = art3d.norm_angle(self.elev - (dy/h)*180)
            self.azim = art3d.norm_angle(self.azim - (dx/w)*180)
            self.get_proj()
            self.figure.canvas.draw()

#        elif self.button_pressed == 2:
            # pan view
            # project xv,yv,zv -> xw,yw,zw
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
            self.figure.canvas.draw()

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        '''Set xlabel.'''

        label = self.w_xaxis.get_label()
        label.set_text(xlabel)
        if fontdict is not None:
            label.update(fontdict)
        label.update(kwargs)
        return label

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        '''Set ylabel.'''

        label = self.w_yaxis.get_label()
        label.set_text(ylabel)
        if fontdict is not None:
            label.update(fontdict)
        label.update(kwargs)
        return label

    def set_zlabel(self, zlabel, fontdict=None, **kwargs):
        '''Set zlabel.'''

        label = self.w_zaxis.get_label()
        label.set_text(zlabel)
        if fontdict is not None:
            label.update(fontdict)
        label.update(kwargs)
        return label

    def grid(self, on=True, **kwargs):
        '''
        Set / unset 3D grid.
        '''
        self._draw_grid = on

    def text(self, x, y, z, s, zdir=None, **kwargs):
        '''
        Add text to the plot. kwargs will be passed on to Axes.text,
        except for the `zdir` keyword, which sets the direction to be
        used as the z direction.
        '''
        text = Axes.text(self, x, y, s, **kwargs)
        art3d.text_2d_to_3d(text, z, zdir)
        return text

    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, **kwargs):
        '''
        Plot 2D or 3D data.

        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *xs*, *ys*  X, y coordinates of vertices

        *zs*        z value(s), either one for all points or one for
                    each point.
        *zdir*      Which direction to use as z ('x', 'y' or 'z')
                    when plotting a 2d set.
        ==========  ================================================

        Other arguments are passed on to
        :func:`~matplotlib.axes.Axes.plot`
        '''

        had_data = self.has_data()
        zs = kwargs.pop('zs', 0)
        zdir = kwargs.pop('zdir', 'z')

        argsi = 0
        # First argument is array of zs
        if len(args) > 0 and cbook.iterable(args[0]) and \
                len(xs) == len(args[0]) and cbook.is_scalar(args[0][0]):
            zs = args[argsi]
            argsi += 1

        # First argument is z value
        elif len(args) > 0 and cbook.is_scalar(args[0]):
            zs = args[argsi]
            argsi += 1

        # Match length
        if not cbook.iterable(zs):
            zs = np.ones(len(xs)) * zs

        lines = Axes.plot(self, xs, ys, *args[argsi:], **kwargs)
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir)

        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines

    plot3D = plot

    def plot_surface(self, X, Y, Z, *args, **kwargs):
        '''
        Create a surface plot.

        By default it will be colored in shades of a solid color,
        but it also supports color mapping by supplying the *cmap*
        argument.

        ============= ================================================
        Argument      Description
        ============= ================================================
        *X*, *Y*, *Z* Data values as numpy.arrays
        *rstride*     Array row stride (step size)
        *cstride*     Array column stride (step size)
        *color*       Color of the surface patches
        *cmap*        A colormap for the surface patches.
        *facecolors*  Face colors for the individual patches
        *norm*        An instance of Normalize to map values to colors
        *vmin*        Minimum value to map
        *vmax*        Maximum value to map
        *shade*       Whether to shade the facecolors
        ============= ================================================

        Other arguments are passed on to
        :func:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection.__init__`
        '''

        had_data = self.has_data()

        rows, cols = Z.shape
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)
        rstride = kwargs.pop('rstride', 10)
        cstride = kwargs.pop('cstride', 10)

        if 'facecolors' in kwargs:
            fcolors = kwargs.pop('facecolors')
        else:
            color = np.array(colorConverter.to_rgba(kwargs.pop('color', 'b')))
            fcolors = None

        cmap = kwargs.get('cmap', None)
        norm = kwargs.pop('norm', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        linewidth = kwargs.get('linewidth', None)
        shade = kwargs.pop('shade', cmap is None)
        lightsource = kwargs.pop('lightsource', None)

        # Shade the data
        if shade and cmap is not None and fcolors is not None:
            fcolors = self._shade_colors_lightsource(Z, cmap, lightsource)

        polys = []
        normals = []
        #colset contains the data for coloring: either average z or the facecolor
        colset = []
        for rs in np.arange(0, rows-1, rstride):
            for cs in np.arange(0, cols-1, cstride):
                ps = []
                corners = []
                for a, ta in [(X, tX), (Y, tY), (Z, tZ)]:
                    ztop = a[rs][cs:min(cols, cs+cstride+1)]
                    zleft = ta[min(cols-1, cs+cstride)][rs:min(rows, rs+rstride+1)]
                    zbase = a[min(rows-1, rs+rstride)][cs:min(cols, cs+cstride+1):]
                    zbase = zbase[::-1]
                    zright = ta[cs][rs:min(rows, rs+rstride+1):]
                    zright = zright[::-1]
                    corners.append([ztop[0], ztop[-1], zbase[0], zbase[-1]])
                    z = np.concatenate((ztop, zleft, zbase, zright))
                    ps.append(z)

                # The construction leaves the array with duplicate points, which
                # are removed here.
                ps = zip(*ps)
                lastp = np.array([])
                ps2 = []
                avgzsum = 0.0
                for p in ps:
                    if p != lastp:
                        ps2.append(p)
                        lastp = p
                        avgzsum += p[2]
                polys.append(ps2)

                if fcolors is not None:
                    colset.append(fcolors[rs][cs])
                else:
                    colset.append(avgzsum / len(ps2))

                # Only need vectors to shade if no cmap
                if cmap is None and shade:
                    v1 = np.array(ps2[0]) - np.array(ps2[1])
                    v2 = np.array(ps2[2]) - np.array(ps2[0])
                    normals.append(np.cross(v1, v2))

        polyc = art3d.Poly3DCollection(polys, *args, **kwargs)

        if fcolors is not None:
            if shade:
                colset = self._shade_colors(colset, normals)
            polyc.set_facecolors(colset)
            polyc.set_edgecolors(colset)
        elif cmap:
            colset = np.array(colset)
            polyc.set_array(colset)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            if shade:
                colset = self._shade_colors(color, normals)
            else:
                colset = color
            polyc.set_facecolors(colset)

        self.add_collection(polyc)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return polyc

    def _generate_normals(self, polygons):
        '''
        Generate normals for polygons by using the first three points.
        This normal of course might not make sense for polygons with
        more than three points not lying in a plane.
        '''

        normals = []
        for verts in polygons:
            v1 = np.array(verts[0]) - np.array(verts[1])
            v2 = np.array(verts[2]) - np.array(verts[0])
            normals.append(np.cross(v1, v2))
        return normals

    def _shade_colors(self, color, normals):
        '''
        Shade *color* using normal vectors given by *normals*.
        *color* can also be an array of the same length as *normals*.
        '''

        shade = []
        for n in normals:
            n = n / proj3d.mod(n)
            shade.append(np.dot(n, [-1, -1, 0.5]))

        shade = np.array(shade)
        mask = ~np.isnan(shade)

        if len(shade[mask]) > 0:
            norm = Normalize(min(shade[mask]), max(shade[mask]))
            if art3d.iscolor(color):
                color = color.copy()
                color[3] = 1
                colors = [color * (0.5 + norm(v) * 0.5) for v in shade]
            else:
                colors = [np.array(colorConverter.to_rgba(c)) * \
                            (0.5 + norm(v) * 0.5) \
                            for c, v in zip(color, shade)]
        else:
            colors = color.copy()

        return colors

    def _shade_colors_lightsource(self, data, cmap, lightsource):
        if lightsource is None:
            lightsource = LightSource(azdeg=135, altdeg=55)
        return lightsource.shade(data, cmap)

    def plot_wireframe(self, X, Y, Z, *args, **kwargs):
        '''
        Plot a 3D wireframe.

        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *X*, *Y*,   Data values as numpy.arrays
        *Z*
        *rstride*   Array row stride (step size)
        *cstride*   Array column stride (step size)
        ==========  ================================================

        Keyword arguments are passed on to
        :func:`matplotlib.collections.LineCollection.__init__`.

        Returns a :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`
        '''

        rstride = kwargs.pop("rstride", 1)
        cstride = kwargs.pop("cstride", 1)

        had_data = self.has_data()
        rows, cols = Z.shape

        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        rii = [i for i in range(0, rows, rstride)]+[rows-1]
        cii = [i for i in range(0, cols, cstride)]+[cols-1]
        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = [zip(xl, yl, zl) for xl, yl, zl in \
                zip(xlines, ylines, zlines)]
        lines += [zip(xl, yl, zl) for xl, yl, zl in \
                zip(txlines, tylines, tzlines)]

        linec = art3d.Line3DCollection(lines, *args, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return linec

    def _3d_extend_contour(self, cset, stride=5):
        '''
        Extend a contour in 3D by creating
        '''

        levels = cset.levels
        colls = cset.collections
        dz = (levels[1] - levels[0]) / 2

        for z, linec in zip(levels, colls):
            topverts = art3d.paths_to_3d_segments(linec.get_paths(), z - dz)
            botverts = art3d.paths_to_3d_segments(linec.get_paths(), z + dz)

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

                v1 = np.array(topverts[0][i1]) - np.array(topverts[0][i2])
                v2 = np.array(topverts[0][i1]) - np.array(botverts[0][i1])
                normals.append(np.cross(v1, v2))

            colors = self._shade_colors(color, normals)
            colors2 = self._shade_colors(color, normals)
            polycol = art3d.Poly3DCollection(polyverts,
                                             facecolors=colors,
                                             edgecolors=colors2)
            polycol.set_sort_zpos(z)
            self.add_collection3d(polycol)

        for col in colls:
            self.collections.remove(col)

    def contour(self, X, Y, Z, *args, **kwargs):
        '''
        Create a 3D contour plot.

        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *X*, *Y*,   Data values as numpy.arrays
        *Z*
        *extend3d*  Whether to extend contour in 3D (default: False)
        *stride*    Stride (step size) for extending contour
        *zdir*      The direction to use: x, y or z (default)
        *offset*    If specified plot a projection of the contour
                    lines on this position in plane normal to zdir
        ==========  ================================================

        The positional and other keyword arguments are passed on to
        :func:`~matplotlib.axes.Axes.contour`

        Returns a :class:`~matplotlib.axes.Axes.contour`
        '''

        extend3d = kwargs.pop('extend3d', False)
        stride = kwargs.pop('stride', 5)
        zdir = kwargs.pop('zdir', 'z')
        offset = kwargs.pop('offset', None)

        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = Axes.contour(self, jX, jY, jZ, *args, **kwargs)

        zdir = '-' + zdir
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            for z, linec in zip(cset.levels, cset.collections):
                if offset is not None:
                    z = offset
                art3d.line_collection_2d_to_3d(linec, z, zdir=zdir)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contour3D = contour

    def contourf(self, X, Y, Z, *args, **kwargs):
        '''
        Plot filled 3D contours.

        *X*, *Y*, *Z*: data points.

        The positional and keyword arguments are passed on to
        :func:`~matplotlib.axes.Axes.contourf`

        Returns a :class:`~matplotlib.axes.Axes.contourf`
        '''

        had_data = self.has_data()

        cset = Axes.contourf(self, X, Y, Z, *args, **kwargs)
        levels = cset.levels
        colls = cset.collections
        for z1, z2, linec in zip(levels, levels[1:], colls):
            art3d.poly_collection_2d_to_3d(linec, z1)
            linec.set_sort_zpos(z1)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contourf3D = contourf

    def add_collection3d(self, col, zs=0, zdir='z'):
        '''
        Add a 3d collection object to the plot.

        2D collection types are converted to a 3D version by
        modifying the object and adding z coordinate information.

        Supported are:
            - PolyCollection
            - LineColleciton
            - PatchCollection
        '''

        if type(col) is collections.PolyCollection:
            art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(min(zs))
        elif type(col) is collections.LineCollection:
            art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(min(zs))
        elif type(col) is collections.PatchCollection:
            art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(min(zs))

        Axes.add_collection(self, col)

    def scatter(self, xs, ys, zs=0, zdir='z', *args, **kwargs):
        '''
        Create a scatter plot.

        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *xs*, *ys*  Positions of data points.
        *zs*        Either an array of the same length as *xs* and
                    *ys* or a single value to place all points in
                    the same plane. Default is 0.
        *zdir*      Which direction to use as z ('x', 'y' or 'z')
                    when plotting a 2d set.
        ==========  ================================================

        Keyword arguments are passed on to
        :func:`~matplotlib.axes.Axes.scatter`.

        Returns a :class:`~mpl_toolkits.mplot3d.art3d.Patch3DCollection`
        '''

        had_data = self.has_data()

        patches = Axes.scatter(self, xs, ys, *args, **kwargs)
        if not cbook.iterable(zs):
            is_2d = True
            zs = np.ones(len(xs)) * zs
        else:
            is_2d = False
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir)

        #FIXME: why is this necessary?
        if not is_2d:
            self.auto_scale_xyz(xs, ys, zs, had_data)

        return patches

    scatter3D = scatter

    def bar(self, left, height, zs=0, zdir='z', *args, **kwargs):
        '''
        Add 2D bar(s).

        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *left*      The x coordinates of the left sides of the bars.
        *height*    The height of the bars.
        *zs*        Z coordinate of bars, if one value is specified
                    they will all be placed at the same z.
        *zdir*      Which direction to use as z ('x', 'y' or 'z')
                    when plotting a 2d set.
        ==========  ================================================

        Keyword arguments are passed onto :func:`~matplotlib.axes.Axes.bar`.

        Returns a :class:`~mpl_toolkits.mplot3d.art3d.Patch3DCollection`
        '''

        had_data = self.has_data()

        patches = Axes.bar(self, left, height, *args, **kwargs)

        if not cbook.iterable(zs):
            zs = np.ones(len(left)) * zs

        verts = []
        verts_zs = []
        for p, z in zip(patches, zs):
            vs = art3d.get_patch_verts(p)
            verts += vs.tolist()
            verts_zs += [z] * len(vs)
            art3d.patch_2d_to_3d(p, zs, zdir)
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])

        xs, ys = zip(*verts)
        xs, ys, verts_zs = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)

        return patches

    def bar3d(self, x, y, z, dx, dy, dz, color='b',
              zsort='average', *args, **kwargs):
        '''
        Generate a 3D bar, or multiple bars.

        When generating multiple bars, x, y, z have to be arrays.
        dx, dy, dz can be arrays or scalars.

        *color* can be:

         - A single color value, to color all bars the same color.

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

        Keyword arguments are passed onto
        :func:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
        '''
        had_data = self.has_data()

        if not cbook.iterable(x):
            x = [x]
        if not cbook.iterable(y):
            y = [y]
        if not cbook.iterable(z):
            z = [z]

        if not cbook.iterable(dx):
            dx = [dx]
        if not cbook.iterable(dy):
            dy = [dy]
        if not cbook.iterable(dz):
            dz = [dz]

        if len(dx) == 1:
            dx = dx * len(x)
        if len(dy) == 1:
            dy = dy * len(y)
        if len(dz) == 1:
            dz = dz * len(z)

        if len(x) != len(y) or len(x) != len(z):
            warnings.warn('x, y, and z must be the same length.')

        minx, miny, minz = 1e20, 1e20, 1e20
        maxx, maxy, maxz = -1e20, -1e20, -1e20

        polys = []
        for xi, yi, zi, dxi, dyi, dzi in zip(x, y, z, dx, dy, dz):
            minx = min(xi, minx)
            maxx = max(xi + dxi, maxx)
            miny = min(yi, miny)
            maxy = max(yi + dyi, maxy)
            minz = min(zi, minz)
            maxz = max(zi + dzi, maxz)

            polys.extend([
                ((xi, yi, zi), (xi + dxi, yi, zi),
                    (xi + dxi, yi + dyi, zi), (xi, yi + dyi, zi)),
                ((xi, yi, zi + dzi), (xi + dxi, yi, zi + dzi),
                    (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

                ((xi, yi, zi), (xi + dxi, yi, zi),
                    (xi + dxi, yi, zi + dzi), (xi, yi, zi + dzi)),
                ((xi, yi + dyi, zi), (xi + dxi, yi + dyi, zi),
                    (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

                ((xi, yi, zi), (xi, yi + dyi, zi),
                    (xi, yi + dyi, zi + dzi), (xi, yi, zi + dzi)),
                ((xi + dxi, yi, zi), (xi + dxi, yi + dyi, zi),
                    (xi + dxi, yi + dyi, zi + dzi), (xi + dxi, yi, zi + dzi)),
            ])

        facecolors = []
        if color is None:
            # no color specified
            facecolors = [None] * len(x)
        elif len(color) == len(x):
            # bar colors specified, need to expand to number of faces
            for c in color:
                facecolors.extend([c] * 6)
        else:
            # a single color specified, or face colors specified explicitly
            facecolors = list(colorConverter.to_rgba_array(color))
            if len(facecolors) < len(x):
                facecolors *= (6 * len(x))

        normals = self._generate_normals(polys)
        sfacecolors = self._shade_colors(facecolors, normals)
        col = art3d.Poly3DCollection(polys,
                                     zsort=zsort,
                                     facecolor=sfacecolors,
                                     *args, **kwargs)
        self.add_collection(col)

        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

    def set_title(self, label, fontdict=None, **kwargs):
        Axes.set_title(self, label, fontdict, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)

def get_test_data(delta=0.05):
    '''
    Return a tuple X, Y, Z with a test data set.
    '''

    from matplotlib.mlab import  bivariate_normal
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z



########################################################
# Register Axes3D as a 'projection' object available 
# for use just like any other axes
########################################################
import matplotlib.projections as proj
proj.projection_registry.register(Axes3D)
