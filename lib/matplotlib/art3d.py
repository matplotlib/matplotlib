#!/usr/bin/python
# art3d.py
#
"""
Wrap 2D artists so that they can pretend to be 3D
"""

import lines
from collections import LineCollection, PolyCollection
import text

from colors import Normalize
from cm import jet

import numpy as npy
import proj3d

class Wrap2D:
    """Wrapper which wraps a 2D object and makes it 3D

    Artists are normally rendered by calling the draw method, this class
    causes call_draw3d to be called instead.
    This in turn calls
    draw3d which should play with the 2D coordinates and eventually
    call the original self.draw method through self.orig_draw.

    overrides the draw method with draw3d
    remembers the original draw method of the wrapped 2d instance
    """
    def __init__(self, inst2d):
        self.__dict__['_wrapped'] = inst2d
        self.__dict__['remembered'] = {}
        #
        inst2d.orig_draw = inst2d.draw
        self.draw2d = inst2d.draw
        inst2d.draw = self.call_draw3d

    def remember(self, *attrs):
        """
        Remember some attributes in the wrapped class
        """
        for attr in attrs:
            assert(hasattr(self._wrapped, attr))
            self.remembered[attr] = 1

    def __getattr__(self, k):
        return getattr(self.__dict__['_wrapped'], k)

    def __setattr__(self, k, v):
        setattr(self.__dict__['_wrapped'], k, v)

    def call_draw3d(self, renderer):
        for k in self.remembered.keys():
            self.remembered[k] = getattr(self, k)
        #
        self.draw3d(renderer)
        #
        for k in self.remembered.keys():
            setattr(self, k, self.remembered[k])

    def draw3d(self, renderer):
        raise ValueError, "draw3d should be overridden"

class Text3DW(Wrap2D):
    """Wrap a 2D text object and make it look vaguely 3D"""
    def __init__(self, inst, z=0, dir='z'):
        Wrap2D.__init__(self, inst)
        self._z = z
        self.dir = dir

    def draw3d(self, renderer):
        x,y = self.get_position()
        xs,ys,zs = juggle_axes(x,y,self._z,self.dir)
        xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
        self.set_x(xs)
        self.set_y(ys)
        #text.Text.draw(self._wrapped, renderer)
        self.draw2d(renderer)
        self.set_x(x)
        self.set_y(y)

class Text3D(Text3DW):
    def __init__(self, x=0,y=0,z=0,text='', dir='z'):
        inst = text.Text(x,y,text,*args, **kwargs)
        Text3DW.__init__(self, z,dir, inst)

class oText3D(text.Text):
    def __init__(self, x=0,y=0,z=0,text='', dir='z', *args, **kwargs):
        text.Text.__init__(self, x,y,text,*args,**kwargs)
        self.dir = dir
        self._z = _z

    def draw(self, renderer):
        x,y = self.get_position()
        xs,ys,zs = juggle_axes(x,y,self._z,self.dir)
        xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
        self.set_x(xs)
        self.set_y(ys)
        text.Text.draw(self, renderer)
        self.set_x(x)
        self.set_y(y)

class Line3D(lines.Line2D):
    """Make a 2D line pretend to be 3D"""
    def __init__(self, xs,ys,zs, *args, **kwargs):
        lines.Line2D.__init__(self, xs,ys, *args, **kwargs)
        self.xs,self.ys,self.zs = xs,ys,zs

    def draw(self, renderer):
        xs,ys,zs = proj3d.proj_transform(self.xs,self.ys,self.zs, renderer.M)
        self._x,self._y = xs,ys

        lines.Line2D.draw(self, renderer)

class Line3DCollectionW(Wrap2D):
    def __init__(self, inst, segments):
        Wrap2D.__init__(self, inst)
        self.segments_3d = segments

    def draw3d(self, renderer):
        xyslist = [
            proj3d.proj_trans_points(points, renderer.M) for points in
            self.segments_3d]
        segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
        self._segments = segments_2d
        self.draw2d(renderer)

class Line3DCollection(Line3DCollectionW):
    def __init__(self, segments, *args, **kwargs):
        inst = LineCollection(segments, *args, **kwargs)
        Line3DCollectionW.__init__(self, inst, segments)

class Line2DCollectionW(Wrap2D):
    def __init__(self, inst, z=0, dir='z'):
        Wrap2D.__init__(self, inst)
        self.z = z
        self.dir = dir
        self.remember('_segments')

    def draw3d(self, renderer):
        #
        segments_3d = [[juggle_axes(x,y,self.z,self.dir) for (x,y) in points]
                       for points in self._segments]
        xyslist = [
            proj3d.proj_trans_points(points, renderer.M) for points in
            segments_3d]
        segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
        #orig_segments = self._segments
        self._segments = segments_2d
        self.draw2d(renderer)
        #self._segments = orig_segments

class Line3DCollection(Line3DCollectionW):
    def __init__(self, segments, *args, **kwargs):
        inst = LineCollection(segments, *args, **kwargs)
        Line3DCollectionW.__init__(self, inst, segments)

class Patch3D(Wrap2D):
    def __init__(self, inst, zs, dir='z'):
        Wrap2D.__init__(self, inst)
        self.zs = zs
        self.dir = dir
        self.remember('get_verts')

    def draw3d(self, renderer):
        xs,ys = zip(*self.get_verts())
        xs,ys,zs = juggle_axes(xs,ys,self.zs,self.dir)
        vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
        def get_verts(*args):
            verts = zip(vxs,vys)
            return verts

        self.get_verts = get_verts
        self.draw2d(renderer)

class Patch3DCollectionW(Wrap2D):
    def __init__(self, inst, zs, dir='z'):
        Wrap2D.__init__(self, inst)
        self.zs = zs
        self.dir = dir
        self.remember('_offsets','_facecolors','_edgecolors')

    def draw3d(self, renderer):
        xs,ys = zip(*self._offsets)
        xs,ys,zs = juggle_axes(xs,ys,self.zs,self.dir)
        if 0:
            vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
            # mess with colors
            #
            vxs = [x for x,i in zip(vxs,vis) if i]
            vys = [y for y,i in zip(vys,vis) if i]
        else:
            vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
        self._facecolors = zalpha(self._facecolors,vzs)
        self._edgecolors = zalpha(self._edgecolors,vzs)
        self._offsets = zip(vxs,vys)
        self.draw2d(renderer)

class Poly3DCollectionW(Wrap2D):
    def __init__(self, inst, zs=None, dir='z'):
        Wrap2D.__init__(self, inst)
        if not zs:
            zs = [0 for v in inst._verts]
        self.zs = [[z for v in verts] for z,verts in zip(zs,inst._verts)]
        self.dir = dir
        self.remember('_verts','_facecolors','_edgecolors')

    def draw3d(self, renderer):
        vverts = []
        for zs,verts in zip(self.zs,self._verts):
            xs,ys = zip(*verts)
            xs,ys,zs = juggle_axes(xs,ys,zs,self.dir)
            vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
            vverts.append((max(vzs),zip(vxs,vys)))
        vverts.sort()
        vverts.reverse()
        # mess with colors
        self._verts = [verts for (z,verts) in vverts]
        self.draw2d(renderer)

class oLine3DCollection(LineCollection):
    def __init__(self, segments, *args, **kwargs):
        LineCollection.__init__(self, segments, *args, **kwargs)
        self.segments_3d = segments

    def draw(self, renderer):
        orig_segments = self._segments
        xyslist = [
            proj3d.proj_trans_points(points, renderer.M) for points in
            self.segments_3d]
        segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
        self._segments = segments_2d
        LineCollection.draw(self, renderer)
        self._segments = orig_segments

class Poly3DCollection(Wrap2D):
    def __init__(self, segments, *args, **kwargs):
        inst = PolyCollection(segments, *args, **kwargs)
        Wrap2D.__init__(self, inst)
        self._zsort = 1
        self.get_vector()
        self.remember('_facecolors')
        self.remember('_verts')

    def get_vector(self):
        """optimise points for projection"""
        si = 0
        ei = 0
        segis = []
        points = []
        for p in self._verts:
            points.extend(p)
            ei = si+len(p)
            segis.append((si,ei))
            si = ei
        xs,ys,zs = zip(*points)
        ones = npy.ones(len(xs))
        self.vec = npy.array([xs,ys,zs,ones])
        self.segis = segis

    def draw3d(self, renderer):
        #
        txs,tys,tzs,tis = proj3d.proj_transform_vec_clip(self.vec,renderer.M)
        xyslist = [(txs[si:ei],tys[si:ei],tzs[si:ei],tis[si:ei]) for si,ei in self.segis]
        colors = get_colors(self._facecolors, len(self._verts))
        #
        # if required sort by depth (furthest drawn first)
        if self._zsort:
            z_segments_2d = [(min(zs),max(tis),zip(xs,ys),c) for
                             (xs,ys,zs,tis),c in zip(xyslist,colors)]
            z_segments_2d.sort()
            z_segments_2d.reverse()
        else:
            raise ValueError, "whoops"
        segments_2d = [s for z,i,s,c in z_segments_2d if i]
        colors = [c for z,i,s,c in z_segments_2d if i]
        self._verts = segments_2d
        self._facecolors = colors

        self.draw2d(renderer)

def juggle_axes(xs,ys,zs, dir):
    """Depending on the direction of the plot re-order the axis

    This is so that 2d plots can be plotted along any direction.
    """
    if dir == 'x': return zs,xs,ys
    elif dir == 'y': return xs,zs,ys
    else: return xs,ys,zs

class Line2DW(Wrap2D):
    def __init__(self, inst, z=0, dir='z'):
        Wrap2D.__init__(self, inst)
        self.z = z
        self.dir = dir
        self.remember('_x','_y')

    def draw3d(self, renderer):
        zs = [self.z for x in self._x]
        xs,ys,zs = juggle_axes(self._x,self._y,zs,self.dir)
        xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
        self._x = xs
        self._y = ys
        self.draw2d(renderer)

def line_draw(self, renderer):
    """Draw a 2D line as a 3D line"""
    oxs,oys = self.get_xdata(),self.get_ydata()
    xs,ys,zs = juggle_axes(oxs,oys,self.zs,self.dir)
    xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
    self._x = xs
    self._y = ys
    self.old_draw(renderer)
    self._x = oxs
    self._y = oys

def wrap_line(line, zs,dir='z'):
    """Wrap a 2D line so that it draws as a 3D line"""
    line.zs = zs
    line.dir = dir
    line.old_draw = line.draw
    def wrapped_draw(renderer,line=line):
        return line_draw(line,renderer)
    line.draw = wrapped_draw

def image_draw(image,renderer):
    source = image._A
    w,h,p = source.shape
    X,Y = meshgrid(arange(w),arange(h))
    Z = npy.zeros((w,h))
    tX,tY,tZ = proj3d.transform(X.flat,Y.flat,Z.flat,M)
    tX = reshape(tX,(w,h))
    tY = reshape(tY,(w,h))

def wrap_image(image, extent):
    image.extent3D = extent
    image.old_draw = image.draw
    def wrapped_draw(renderer,image=image):
        return image_draw(image,renderer)
    image.draw = wrapped_draw


def set_line_data(line, xs,ys,zs):
    try: line = line[0]
    except: pass
    line.set_data(xs,ys)
    line.zs = zs

def iscolor(c):
    try:
        return (len(c)==4 or len(c)==3) and (type(c[0])==float)
    except (IndexError):
        return None

def get_colors(c, num):
    """Stretch the color argument to provide the required number num"""
    if type(c)==type("string"):
        c = colors.colorConverter.to_rgba(colors)
    if iscolor(c):
        return [c]*num
    elif iscolor(c[0]):
        return c*num
    elif len(c)==num:
        return c[:]
    else:
        raise ValueError, 'unknown color format %s' % c

def zalpha(colors, zs):
    """Modify the alphas of the color list according to depth"""
    colors = get_colors(colors,len(zs))
    norm = Normalize(min(zs),max(zs))
    sats = 1 - norm(zs)*0.7
    colors = [(c[0],c[1],c[2],c[3]*s) for c,s in zip(colors,sats)]
    return colors

def patch_draw(self, renderer):
    orig_offsets = self._offsets
    xs,ys = zip(*self._offsets)
    xs,ys,zs = juggle_axes(xs,ys,self.zs,self.dir)
    xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
    # mess with colors
    orig_fcolors = self._facecolors
    orig_ecolors = self._edgecolors
    self._facecolors = zalpha(orig_fcolors,zs)
    self._edgecolors = zalpha(orig_ecolors,zs)

    self._offsets = zip(xs,ys)
    self.old_draw(renderer)
    self._offsets = orig_offsets
    self._facecolors = orig_fcolors
    self._edgecolors = orig_ecolors

def wrap_patch(patch, zs, dir='z'):
    return Patch3DCollectionW(patch, zs, dir)

def draw_linec(self, renderer):
    orig_segments = self._segments
    segments_3d = [[(x,y,z) for (x,y),z in zip(points,zs)]
                   for zs, points in zip(self.zs, self._segments)]
    xyslist = [
        proj3d.proj_trans_points(points, renderer.M) for points in
        segments_3d]
    segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
    self._segments = segments_2d
    LineCollection.draw(self, renderer)
    self._segments = orig_segments

def draw_polyc(self, renderer):
    orig_segments = self._verts
    # process the list of lists of 2D points held in _verts to generate
    # a list of lists of 3D points
    segments_3d = [[(x,y,z) for (x,y),z in zip(points,self.zs)]
                   for points in self._verts]
    #
    xyslist = [
        proj3d.proj_trans_points(points, renderer.M) for points in
        segments_3d]
    segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
    self._verts = segments_2d
    PolyCollection.draw(self, renderer)
    self._verts = orig_segments

def text_draw(self, renderer):
    x,y = self.get_position()
    xs,ys,zs = juggle_axes(x,y,self._z,self.dir)
    xs,ys,zs = proj3d.proj_transform(xs,ys,zs, renderer.M)
    self.set_x(xs)
    self.set_y(ys)
    self.old_draw(renderer)
    self.set_x(x)
    self.set_y(y)

def wrap_text(text, zs, dir='z'):
    text._z = zs
    text.dir = dir
    text.old_draw = text.draw
    def wrapped_draw(renderer,text=text):
        return text_draw(text,renderer)
    text.draw = wrapped_draw

def set_text_data(text, x,y,z):
    text._x,text._y,text._z = x,y,z

def draw(text, renderer):
    print 'call draw text', text
    print text.get_visible()
    print 'text "%s"' % text._text
    res = text._get_layout(renderer)
    print res
    text._draw(renderer)

def owrap(text):
    text._draw = text.draw
    def draw_text(renderer,text=text):
        draw(text,renderer)
    text.draw = draw_text

def wrap_2d_fn(patch, zs,dir='z',fn=patch_draw):
    patch.zs = zs
    patch.dir = dir
    patch.old_draw = patch.draw
    def wrapped_draw(renderer,patch=patch,fn=fn):
        return fn(patch,renderer)
    patch.draw = wrapped_draw
    return patch
