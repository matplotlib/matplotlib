#!/usr/bin/python
# axes3d.py
#
# Created: 23 Sep 2005
#
"""
3D projection glued onto 2D Axes.

Axes3D
"""

import random

import matplotlib
from axes import Axes
import cbook
from transforms import unit_bbox

import figure
import numpy as npy
from colors import Normalize

import art3d
import proj3d
import axis3d

def sensible_format_data(self, value):
    """Used to generate more comprehensible numbers in status bar"""
    if abs(value) > 1e4 or abs(value)<1e-3:
        s = '%1.4e'% value
        return self._formatSciNotation(s)
    else:
        return '%4.3f' % value

class Axes3DI(Axes):
    """Wrap an Axes object

    The x,y data coordinates, which are manipulated by set_xlim and
    set_ylim are used as the target view coordinates by the 3D
    transformations. These coordinates are mostly invisible to the
    outside world.

    set_w_xlim, set_w_ylim and set_w_zlim manipulate the 3D world
    coordinates which are scaled to represent the data and are stored
    in the xy_dataLim, zz_datalim bboxes.

    The axes representing the x,y,z world dimensions are self.w_xaxis,
    self.w_yaxis and self.w_zaxis. They can probably be controlled in
    more or less the normal ways.
    """
    def __init__(self, fig, rect=[0.0, 0.0, 1.0, 1.0], *args, **kwargs):
        self.fig = fig

        azim = cbook.popd(kwargs, 'azim', -60)
        elev = cbook.popd(kwargs, 'elev', 30)

        self.xy_viewLim = unit_bbox()
        self.zz_viewLim = unit_bbox()
        self.xy_dataLim = unit_bbox()
        self.zz_dataLim = unit_bbox()
        # inihibit autoscale_view until the axises are defined
        # they can't be defined until Axes.__init__ has been called
        self._ready = 0
        Axes.__init__(self, self.fig, rect,
                      frameon=True,
                      xticks=[], yticks=[], *args, **kwargs)

        self.M = None
        self._ready = 1

        self.view_init(elev, azim)
        self.mouse_init()
        self.create_axes()
        self.set_top_view()

        #self.axesPatch.set_edgecolor((1,0,0,0))
        self.axesPatch.set_linewidth(0)
        #self.axesPatch.set_facecolor((0,0,0,0))
        self.fig.add_axes(self)

    def set_top_view(self):
        # this happens to be the right view for the viewing coordinates
        # moved up and to the left slightly to fit labels and axes
        xdwl = (0.95/self.dist)
        xdw = (0.9/self.dist)
        ydwl = (0.95/self.dist)
        ydw = (0.9/self.dist)
        #
        self.set_xlim(-xdwl,xdw)
        self.set_ylim(-ydwl,ydw)

    def really_set_xlim(self, vmin, vmax):
        self.viewLim.intervalx().set_bounds(vmin, vmax)

    def really_set_ylim(self, vmin, vmax):
        self.viewLim.intervaly().set_bounds(vmin, vmax)

    def vlim_argument(self, get_lim, *args):
        if not args:
            vmin,vmax = get_lim()
        elif len(args)==2:
            vmin,vmax = args
        elif len(args)==1:
            vmin,vmax = args[0]
        return vmin,vmax

    def nset_xlim(self, *args):
        raise
        vmin,vmax = self.vlim_argument(self.get_xlim)
        print 'xlim', vmin,vmax

    def nset_ylim(self, *args):
        vmin,vmax = self.vlim_argument(self.get_ylim)
        print 'ylim', vmin,vmax

    def create_axes(self):
        self.w_xaxis = axis3d.Axis('x',self.xy_viewLim.intervalx,
                            self.xy_dataLim.intervalx, self)
        self.w_yaxis = axis3d.Axis('y',self.xy_viewLim.intervaly,
                            self.xy_dataLim.intervaly, self)
        self.w_zaxis = axis3d.Axis('z',self.zz_viewLim.intervalx,
                            self.zz_dataLim.intervalx, self)

    def unit_cube(self,vals=None):
        minpy,maxx,miny,maxy,minz,maxz = vals or self.get_w_lims()
        xs,ys,zs = ([minpy,maxx,maxx,minpy,minpy,maxx,maxx,minpy],
                    [miny,miny,maxy,maxy,miny,miny,maxy,maxy],
                    [minz,minz,minz,minz,maxz,maxz,maxz,maxz])
        return zip(xs,ys,zs)

    def tunit_cube(self,vals=None,M=None):
        if M is None:
            M = self.M
        xyzs = self.unit_cube(vals)
        tcube = proj3d.proj_points(xyzs,M)
        return tcube

    def tunit_edges(self, vals=None,M=None):
        tc = self.tunit_cube(vals,M)
        edges = [(tc[0],tc[1]),
                 (tc[1],tc[2]),
                 (tc[2],tc[3]),
                 (tc[3],tc[0]),

                 (tc[0],tc[4]),
                 (tc[1],tc[5]),
                 (tc[2],tc[6]),
                 (tc[3],tc[7]),

                 (tc[4],tc[5]),
                 (tc[5],tc[6]),
                 (tc[6],tc[7]),
                 (tc[7],tc[4])]
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

        #self.set_top_view()
        self.w_xaxis.draw(renderer)
        self.w_yaxis.draw(renderer)
        self.w_zaxis.draw(renderer)
        Axes.draw(self, renderer)

    def get_axis_position(self):
        vals = self.get_w_lims()
        tc = self.tunit_cube(vals,self.M)
        xhigh = tc[1][2]>tc[2][2]
        yhigh = tc[3][2]>tc[2][2]
        zhigh = tc[0][2]>tc[2][2]
        return xhigh,yhigh,zhigh

    def update_datalim(self, xys):
        pass

    def update_datalim_numerix(self, x, y):
        pass

    def auto_scale_xyz(self, X,Y,Z=None,had_data=None):
        x,y,z = map(npy.asarray, (X,Y,Z))
        try:
            x,y = X.flat,Y.flat
            if Z is not None:
                z = Z.flat
        except AttributeError:
            pass

        self.xy_dataLim.update_numerix(x, y, not had_data)
        if z is not None:
            self.zz_dataLim.update_numerix(z, z, not had_data)
        self.autoscale_view()

    def autoscale_view(self, scalex=True, scaley=True, scalez=True):
        self.set_top_view()
        if not self._ready: return

        if not self._autoscaleon: return

        if scalex:
            locator = self.w_xaxis.get_major_locator()
            self.set_w_xlim(locator.autoscale())
        if scaley:
            locator = self.w_yaxis.get_major_locator()
            self.set_w_ylim(locator.autoscale())
        if scalez:
            locator = self.w_zaxis.get_major_locator()
            self.set_w_zlim(locator.autoscale())

    def get_w_lims(self):
        minpy,maxx = self.get_w_xlim()
        miny,maxy = self.get_w_ylim()
        minz,maxz = self.get_w_zlim()
        return minpy,maxx,miny,maxy,minz,maxz

    def set_w_zlim(self, *args, **kwargs):
        gl,self.get_xlim = self.get_xlim,self.get_w_zlim
        vl,self.viewLim = self.viewLim,self.zz_viewLim
        vmin,vmax = Axes.set_xlim(self, *args, **kwargs)
        self.get_xlim = gl
        self.viewLim = vl
        return vmin,vmax

    def set_w_xlim(self, *args, **kwargs):
        gl,self.get_xlim = self.get_xlim,self.get_w_xlim
        vl,self.viewLim = self.viewLim,self.xy_viewLim
        vmin,vmax = Axes.set_xlim(self, *args, **kwargs)
        self.get_xlim = gl
        self.viewLim = vl
        return vmin,vmax

    def set_w_ylim(self, *args, **kwargs):
        gl,self.get_ylim = self.get_ylim,self.get_w_ylim
        vl,self.viewLim = self.viewLim,self.xy_viewLim
        vmin,vmax = Axes.set_ylim(self, *args, **kwargs)
        self.get_ylim = gl
        self.viewLim = vl
        return vmin,vmax

    def get_w_zlim(self):
        return self.zz_viewLim.intervalx().get_bounds()

    def get_w_xlim(self):
        return self.xy_viewLim.intervalx().get_bounds()

    def get_w_ylim(self):
        return self.xy_viewLim.intervaly().get_bounds()

    def pany(self, numsteps):
        print 'numsteps', numsteps

    def panpy(self, numsteps):
        print 'numsteps', numsteps

    def view_init(self, elev, azim):
        self.dist = 10
        self.elev = elev
        self.azim = azim

    def get_proj(self):
        """Create the projection matrix from the current viewing
        position.

        elev stores the elevation angle in the z plane
        azim stores the azimuth angle in the x,y plane

        dist is the distance of the eye viewing point from the object
        point.

        """
        relev,razim = npy.pi * self.elev/180, npy.pi * self.azim/180

        xmin,xmax = self.get_w_xlim()
        ymin,ymax = self.get_w_ylim()
        zmin,zmax = self.get_w_zlim()

        # transform to uniform world coordinates 0-1.0,0-1.0,0-1.0
        worldM = proj3d.world_transformation(xmin,xmax,
                                             ymin,ymax,
                                             zmin,zmax)

        # look into the middle of the new coordinates
        R = npy.array([0.5,0.5,0.5])
        #
        xp = R[0] + npy.cos(razim)*npy.cos(relev)*self.dist
        yp = R[1] + npy.sin(razim)*npy.cos(relev)*self.dist
        zp = R[2] + npy.sin(relev)*self.dist

        E = npy.array((xp, yp, zp))
        #
        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / proj3d.mod(self.vvec)

        if abs(relev) > npy.pi/2:
            # upside down
            V = npy.array((0,0,-1))
        else:
            V = npy.array((0,0,1))
        zfront,zback = -self.dist,self.dist

        viewM = proj3d.view_transformation(E,R,V)
        perspM = proj3d.persp_transformation(zfront,zback)
        M0 = npy.dot(viewM,worldM)
        M = npy.dot(perspM,M0)
        return M

    def mouse_init(self):
        self.button_pressed = None
        if self.figure.canvas != None:
            self.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
            self.figure.canvas.mpl_connect('button_press_event', self.button_press)
            self.figure.canvas.mpl_connect('button_release_event', self.button_release)

    def button_press(self, event):
        self.button_pressed = event.button
        self.sx,self.sy = event.xdata,event.ydata

    def button_release(self, event):
        self.button_pressed = None

    def format_xdata(self, x):
        """
        Return x string formatted.  This function will use the attribute
        self.fmt_xdata if it is callable, else will fall back on the xaxis
        major formatter
        """
        try: return self.fmt_xdata(x)
        except TypeError:
            fmt = self.w_xaxis.get_major_formatter()
            return sensible_format_data(fmt,x)

    def format_ydata(self, y):
        """
        Return y string formatted.  This function will use the attribute
        self.fmt_ydata if it is callable, else will fall back on the yaxis
        major formatter
        """
        try: return self.fmt_ydata(y)
        except TypeError:
            fmt = self.w_yaxis.get_major_formatter()
            return sensible_format_data(fmt,y)

    def format_zdata(self, z):
        """
        Return y string formatted.  This function will use the attribute
        self.fmt_ydata if it is callable, else will fall back on the yaxis
        major formatter
        """
        try: return self.fmt_zdata(z)
        except (AttributeError,TypeError):
            fmt = self.w_zaxis.get_major_formatter()
            return sensible_format_data(fmt,z)

    def format_coord(self, xd, yd):
        """Given the 2D view coordinates attempt to guess a 3D coordinate

        Looks for the nearest edge to the point and then assumes that the point is
        at the same z location as the nearest point on the edge.
        """

        if self.button_pressed == 1:
            return 'azimuth=%d deg, elevation=%d deg ' % (self.azim, self.elev)
            # ignore xd and yd and display angles instead

        p = (xd,yd)
        edges = self.tunit_edges()
        #lines = [proj3d.line2d(p0,p1) for (p0,p1) in edges]
        ldists = [(proj3d.line2d_seg_dist(p0,p1,p),i) for i,(p0,p1) in enumerate(edges)]
        ldists.sort()
        # nearest edge
        edgei = ldists[0][1]
        #
        p0,p1 = edges[edgei]

        # scale the z value to match
        x0,y0,z0 = p0
        x1,y1,z1 = p1
        d0 = npy.hypot(x0-xd,y0-yd)
        d1 = npy.hypot(x1-xd,y1-yd)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1
        #print 'mid', edgei, d0, d1, z0, z1, z

        x,y,z = proj3d.inv_transform(xd,yd,z,self.M)

        xs = self.format_xdata(x)
        ys = self.format_ydata(y)
        zs = self.format_ydata(z)
        return  'x=%s, y=%s, z=%s'%(xs,ys,zs)

    def on_move(self, event):
        """Mouse moving

        button-1 rotates
        button-3 zooms
        """
        if not self.button_pressed:
            return

        if self.M is None:
            return
            # this shouldn't be called before the graph has been drawn for the first time!

        x, y = event.xdata, event.ydata
        dx,dy = x-self.sx,y-self.sy
        x0,x1 = self.get_xlim()
        y0,y1 = self.get_ylim()
        w = (x1-x0)
        h = (y1-y0)
        self.sx,self.sy = x,y

        if self.button_pressed == 1:
            # rotate viewing point
            # get the x and y pixel coords
            if dx == 0 and dy == 0: return
            #
            self.elev = axis3d.norm_angle(self.elev - (dy/h)*180)
            self.azim = axis3d.norm_angle(self.azim - (dx/w)*180)
            self.get_proj()
            self.figure.canvas.draw()
        elif self.button_pressed == 2:
            # pan view
            # project xv,yv,zv -> xw,yw,zw
            # pan
            #
            pass
        elif self.button_pressed == 3:
            # zoom view
            # hmmm..this needs some help from clipping....
            minpy,maxx,miny,maxy,minz,maxz = self.get_w_lims()
            df = 1-((h - dy)/h)
            dx = (maxx-minpy)*df
            dy = (maxy-miny)*df
            dz = (maxz-minz)*df
            self.set_w_xlim(minpy-dx,maxx+dx)
            self.set_w_ylim(miny-dy,maxy+dy)
            self.set_w_zlim(minz-dz,maxz+dz)
            self.get_proj()
            self.figure.canvas.draw()

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        #par = cbook.popd(kwargs, 'par',None)
        #label.set_par(par)
        #
        label = self.w_xaxis.get_label()
        label.set_text(xlabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        label = self.w_yaxis.get_label()
        label.set_text(ylabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label

    def set_zlabel(self, zlabel, fontdict=None, **kwargs):
        label = self.w_zaxis.get_label()
        label.set_text(zlabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label

    def plot(self, *args, **kwargs):
        had_data = self.has_data()

        zval = cbook.popd(kwargs, 'z', 0)
        zdir = cbook.popd(kwargs, 'dir', 'z')
        lines = Axes.plot(self, *args, **kwargs)
        #
        linecs = [art3d.Line2DW(l, z=zval, dir=zdir) for l in lines]
        #
        xs = lines[0].get_xdata()
        ys = lines[0].get_ydata()
        zs = [zval for x in xs]
        xs,ys,zs = art3d.juggle_axes(xs,ys,zs,zdir)
        #
        self.auto_scale_xyz(xs,ys,zs, had_data)
        #
        return linecs

    def plot3D(self, xs, ys, zs, *args, **kwargs):
        had_data = self.has_data()
        lines = Axes.plot(self, xs,ys, *args, **kwargs)
        if len(lines)==1:
            line = lines[0]
            art3d.wrap_line(line, zs)
        #
        self.auto_scale_xyz(xs,ys,zs, had_data)
        return lines

    plot3d=plot3D

    def plot_surface(self, X, Y, Z, *args, **kwargs):
        had_data = self.has_data()

        rows, cols = Z.shape
        tX,tY,tZ = npy.transpose(X), npy.transpose(Y), npy.transpose(Z)
        rstride = cbook.popd(kwargs, 'rstride', 10)
        cstride = cbook.popd(kwargs, 'cstride', 10)
        #
        polys = []
        boxes = []
        for rs in npy.arange(0,rows-1,rstride):
            for cs in npy.arange(0,cols-1,cstride):
                ps = []
                corners = []
                for a,ta in [(X,tX),(Y,tY),(Z,tZ)]:
                    ztop = a[rs][cs:min(cols,cs+cstride+1)]
                    zleft = ta[min(cols-1,cs+cstride)][rs:min(rows,rs+rstride+1)]
                    zbase = a[min(rows-1,rs+rstride)][cs:min(cols,cs+cstride+1):]
                    zbase = zbase[::-1]
                    zright = ta[cs][rs:min(rows,rs+rstride+1):]
                    zright = zright[::-1]
                    corners.append([ztop[0],ztop[-1],zbase[0],zbase[-1]])
                    z = npy.concatenate((ztop,zleft,zbase,zright))
                    ps.append(z)
                boxes.append(map(npy.array,zip(*corners)))
                polys.append(zip(*ps))
        #
        lines = []
        shade = []
        for box in boxes:
            n = proj3d.cross(box[0]-box[1],
                         box[0]-box[2])
            n = n/proj3d.mod(n)*5
            shade.append(npy.dot(n,[-1,-1,0.5]))
            lines.append((box[0],n+box[0]))
        #
        color = npy.array([0,0,1,1])
        norm = Normalize(min(shade),max(shade))
        colors = [color * (0.5+norm(v)*0.5) for v in shade]
        for c in colors: c[3] = 1
        polyc = art3d.Poly3DCollection(polys, facecolors=colors, *args, **kwargs)
        polyc._zsort = 1
        self.add_collection(polyc)
        #
        self.auto_scale_xyz(X,Y,Z, had_data)
        return polyc

    def plot_wireframe(self, X, Y, Z, *args, **kwargs):
        rstride = cbook.popd(kwargs, "rstride", 1)
        cstride = cbook.popd(kwargs, "cstride", 1)

        had_data = self.has_data()
        rows,cols = Z.shape

        tX,tY,tZ = npy.transpose(X), npy.transpose(Y), npy.transpose(Z)

        rii = [i for i in range(0,rows,rstride)]+[rows-1]
        cii = [i for i in range(0,cols,cstride)]+[cols-1]
        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]
        #
        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]
        #
        lines = [zip(xl,yl,zl) for xl,yl,zl in zip(xlines,ylines,zlines)]
        lines += [zip(xl,yl,zl) for xl,yl,zl in zip(txlines,tylines,tzlines)]
        linec = self.add_lines(lines, *args, **kwargs)

        self.auto_scale_xyz(X,Y,Z, had_data)
        return linec

    def contour3D(self, X, Y, Z, *args, **kwargs):
        had_data = self.has_data()
        cset = self.contour(X, Y, Z, *args, **kwargs)
        for z,linec in zip(cset.levels,cset.collections):
            zl = []
            linew = art3d.Line2DCollectionW(linec, z)
        self.auto_scale_xyz(X,Y,Z, had_data)
        return cset

    def clabel(self, *args, **kwargs):
        r = Axes.clabel(self, *args, **kwargs)
        return r

    def contourf3D(self, X, Y, Z, *args, **kwargs):
        raise NotImplementedError("contourf3D is broken")
        had_data = self.has_data()

        cset = self.contourf(X, Y, Z, *args, **kwargs)
        levels = cset.levels
        colls = cset.collections
        for z1,z2,linec in zip(levels,levels[1:],colls):
            zs = [z1] * (len(linec._verts[0])/2)
            zs += [z2] * (len(linec._verts[0])/2)
            # The following is clearly wrong.
            art3d.wrap_patch(linec, zs, fn=art3d.draw_polyc)
        self.auto_scale_xyz(X,Y,Z, had_data)
        return cset

    def scatter3D(self, xs, ys, zs, *args, **kwargs):
        had_data = self.has_data()
        patches = Axes.scatter(self,xs,ys,*args,**kwargs)
        patches = art3d.wrap_patch(patches, zs)
        #
        self.auto_scale_xyz(xs,ys,zs, had_data)
        return patches
    scatter3d = scatter3D

    def add_lines(self, lines, *args, **kwargs):
        linec = art3d.Line3DCollection(lines, *args, **kwargs)
        self.add_collection(linec)
        return linec

    def text3D(self, x,y,z,s, *args, **kwargs):
        text = Axes.text(self,x,y,s,*args,**kwargs)
        art3d.wrap_text(text,z)
        return text

    def ahvline(self, x,y):
        pass

    def ahvxplane(self, x):
        pass

    def ahvyplane(self, y):
        pass

class Scaler:
    def __init__(self, points):
        self.inpoints = points
        self.drawpoints = None

    def update(self, lims):
        for x,y,z in self.points:
            pass

class Axes3D:
    """
    Wrapper for Axes3DI

    Provides set_xlim, set_ylim etc.

    2D functions can be caught here and mapped
    to their 3D approximations.

    This should probably be the case for plot etc...
    """
    def __init__(self, fig, *args, **kwargs):
        self.__dict__['wrapped'] = Axes3DI(fig, *args, **kwargs)

    def set_xlim(self, *args, **kwargs):
        self.wrapped.set_w_xlim(*args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        self.wrapped.set_w_ylim(*args, **kwargs)

    def set_zlim(self, *args, **kwargs):
        self.wrapped.set_w_zlim(*args, **kwargs)

    def __getattr__(self, k):
        return getattr(self.wrapped,k)

    def __setattr__(self, k,v):
        return setattr(self.wrapped,k,v)

    def add_collection(self, polys, zs=None, dir='z'):
        patches = art3d.Poly3DCollectionW(polys, zs=zs,dir=dir)
        self.add_3DCollection(patches)

    def add_3DCollection(self, patches):
        self.wrapped.add_collection(patches)

    def text(self, x,y, text, *args,**kwargs):
        self.wrapped.text3D(x,y,0,text,*args,**kwargs)

    def scatter(self, xs,ys,zs=None,dir='z',*args,**kwargs):
        patches = self.wrapped.scatter(xs,ys,*args,**kwargs)
        if zs is None:
            zs = [0]*len(xs)
        patches = art3d.wrap_patch(patches, zs=zs, dir=dir)
        return patches

    def bar(self, left, height, z=0, dir='z', *args, **kwargs):
        had_data = self.has_data()
        patches = self.wrapped.bar(left, height, *args, **kwargs)
        #
        verts = []
        for p in patches:
            vs = p.get_verts()
            zs = [z]*len(vs)
            verts += vs
            patch3d = art3d.Patch3D(p, zs, dir=dir)
        xs,ys = zip(*verts)
        zs = [z]*len(xs)
        xs,ys,zs=art3d.juggle_axes(xs,ys,zs,dir)
        self.wrapped.auto_scale_xyz(xs,ys,zs, had_data)

def test_scatter():

    ax = Axes3D()
    #
    #
    n = 100
    for c,zl,zh in [('r',-50,-25),('b',-30,-5)]:
        xs,ys,zs = zip(*
                       [(random.randrange(23,32),
                         random.randrange(100),
                         random.randrange(zl,zh)
                         ) for i in range(n)])
        ax.scatter3D(xs,ys,zs, c=c)
    #
    ax.set_xlabel('------------ X Label --------------------')
    ax.set_ylabel('------------ Y Label --------------------')
    ax.set_zlabel('------------ Z Label --------------------')

def get_test_data(delta=0.05):
    from mlab import  bivariate_normal
    x = y = npy.arange(-3.0, 3.0, delta)
    X, Y = npy.meshgrid(x,y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2-Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X,Y,Z

def test_wire():
    ax = Axes3D()

    X,Y,Z = get_test_data(0.05)
    ax.plot_wireframe(X,Y,Z, rstride=10,cstride=10)
    #
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_surface():
    ax = Axes3D()

    X,Y,Z = get_test_data(0.05)
    ax.plot_surface(X,Y,Z, rstride=10,cstride=10)
    #
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def test_contour():
    ax = Axes3D()

    X,Y,Z = get_test_data(0.05)
    cset = ax.contour3D(X,Y,Z)
    ax.clabel(cset, fontsize=9, inline=1)
    #
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def test_plot():
    ax = Axes3D()
    xs = npy.arange(0,4*npy.pi+0.1,0.1)
    ys = npy.sin(xs)
    ax.plot(xs,ys, label='zl')
    ax.plot(xs,ys+max(xs),label='zh')
    ax.plot(xs,ys,dir='x', label='xl')
    ax.plot(xs,ys,dir='x', z=max(xs),label='xh')
    ax.plot(xs,ys,dir='y', label='yl')
    ax.plot(xs,ys,dir='y', z=max(xs), label='yh')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


def test_polys():
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import colorConverter

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

    ax = Axes3D()
    xs = npy.arange(0,10,0.4)
    verts = []
    zs = [0.0,1.0,2.0,3.0]
    for z in zs:
        ys = [random.random() for x in xs]
        ys[0],ys[-1] = 0,0
        verts.append(zip(xs,ys))

    poly = PolyCollection(verts, facecolors = [cc('r'),cc('g'),cc('b'),
                                               cc('y')])
    #patches = art3d.Poly3DCollectionW(poly, zs=zs, dir='y')
    #poly = PolyCollection(verts)
    ax.add_collection(poly,zs=zs,dir='y')
    #ax.wrapped.add_collection(poly)
    #
    ax.plot(xs,ys, z=z, dir='y', c='r')
    ax.set_xlim(0,10)
    ax.set_ylim(-1,4)
    ax.set_zlim(0,1)

def test_scatter2D():
    xs = [random.random() for i in range(20)]
    ys = [random.random() for x in xs]
    ax = Axes3D()
    ax.scatter(xs,ys)
    ax.scatter(xs,ys, dir='y', c='r')
    ax.scatter(xs,ys, dir='x', c='g')

def test_bar2D():
    ax = Axes3D()

    for c,z in zip(['r','g','b','y'],[30,20,10,0]):
        xs = npy.arange(20)
        ys = [random.random() for x in xs]
        ax.bar(xs,ys,z=z,dir='y',color=c)
    #ax.plot(xs,ys)

if __name__ == "__main__":
    import pylab
    #test_scatter()
    #test_wire()
    #test_surface()
    #test_contour()
    #test_plot()
    test_polys()
    #test_scatter2D()
    test_bar2D()
    pylab.show()
