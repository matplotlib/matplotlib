#!/usr/bin/python
# art3d.py, original mplot3d version by John Porter
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>

from matplotlib import lines, text, path as mpath
from matplotlib.collections import Collection, LineCollection, \
        PolyCollection, PatchCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize
from matplotlib import transforms

import types
import numpy as np
import proj3d

class Text3D(text.Text):

    def __init__(self, x=0, y=0, z=0, text='', dir='z'):
        text.Text.__init__(self, x, y, text)
        self.set_3d_properties(z, dir)

    def set_3d_properties(self, z=0, dir='z'):
        x, y = self.get_position()
        self._position3d = juggle_axes(x, y, z, dir)

    def draw(self, renderer):
        x, y, z = self._position3d
        x, y, z = proj3d.proj_transform(x, y, z, renderer.M)
        self.set_position(x, y)
        text.Text.draw(self, renderer)

def text_2d_to_3d(obj, z=0, dir='z'):
    """Convert a Text to a Text3D object."""
    obj.__class__ = Text3D
    obj.set_3d_properties(z, dir)

class Line3D(lines.Line2D):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        lines.Line2D.__init__(self, [], [], *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_3d_properties(self, zs=0, dir='z'):
        xs = self.get_xdata()
        ys = self.get_ydata()
        try:
            zs = float(zs)
            zs = [zs for x in xs]
        except:
            pass
        self._verts3d = juggle_axes(xs, ys, zs, dir)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_data(xs, ys)
        lines.Line2D.draw(self, renderer)

def line_2d_to_3d(line, z=0, dir='z'):
    line.__class__ = Line3D
    line.set_3d_properties(z, dir)

def path_to_3d_segment(path, z=0, dir='z'):
    '''Convert a path to a 3d segment.'''
    seg = []
    for (pathseg, code) in path.iter_segments():
        seg.append(pathseg)
    seg3d = [juggle_axes(x, y, z, dir) for (x, y) in seg]
    return seg3d

def paths_to_3d_segments(paths, zs=0, dir='z'):
    '''Convert paths from a collection object to 3d segments.'''

    try:
        zs = float(zs)
        zs = [zs for i in range(len(paths))]
    except:
        pass

    segments = []
    for path, z in zip(paths, zs):
        segments.append(path_to_3d_segment(path, z, dir))
    return segments

class Line3DCollection(LineCollection):

    def __init__(self, segments, *args, **kwargs):
        LineCollection.__init__(self, segments, *args, **kwargs)

    def set_segments(self, segments):
        self._segments3d = segments
        LineCollection.set_segments(self, [])

    def do_3d_projection(self, renderer):
        xyslist = [
            proj3d.proj_trans_points(points, renderer.M) for points in
            self._segments3d]
        segments_2d = [zip(xs,ys) for (xs,ys,zs) in xyslist]
        LineCollection.set_segments(self, segments_2d)

        minz = 1e9
        for (xs, ys, zs) in xyslist:
            minz = min(minz, min(zs))
        return minz

    def draw(self, renderer, project=False):
        if project:
            self.do_3d_projection(renderer)
        LineCollection.draw(self, renderer)

def line_collection_2d_to_3d(col, z=0, dir='z'):
    """Convert a LineCollection to a Line3DCollection object."""
    segments3d = paths_to_3d_segments(col.get_paths(), z, dir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)

class Patch3D(Patch):

    def __init__(self, *args, **kwargs):
        zs = kwargs.pop('zs', [])
        dir = kwargs.pop('dir', 'z')
        Patch.__init__(self, *args, **kwargs)
        self.set_3d_properties(zs, dir)

    def set_3d_properties(self, verts, z=0, dir='z'):
        self._segment3d = [juggle_axes(x, y, z, dir) for (x, y) in verts]
        self._facecolor3d = Patch.get_facecolor(self)

    def get_path(self):
        return self._path2d

    def get_facecolor(self):
        return self._facecolor2d

    def do_3d_projection(self, renderer):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
        self._path2d = mpath.Path(zip(vxs, vys))
        # FIXME: coloring
        self._facecolor2d = self._facecolor3d
        return min(vzs)

    def draw(self, renderer):
        Patch.draw(self, renderer)

def patch_2d_to_3d(patch, z=0, dir='z'):
    """Convert a Patch to a Patch3D object."""
    verts = patch.get_verts()
    patch.__class__ = Patch3D
    patch.set_3d_properties(verts, z, dir)

class Patch3DCollection(PatchCollection):

    def __init__(self, *args, **kwargs):
        PatchCollection.__init__(self, *args, **kwargs)

    def set_3d_properties(self, zs, dir):
        xs, ys = zip(*self.get_offsets())
        self._offsets3d = juggle_axes(xs, ys, zs, dir)
        self._facecolor3d = self.get_facecolor()
        self._edgecolor3d = self.get_edgecolor()

    def do_3d_projection(self, renderer):
        xs,ys,zs = self._offsets3d
        vxs,vys,vzs,vis = proj3d.proj_transform_clip(xs,ys,zs, renderer.M)
        #FIXME: mpl allows us no way to unset the collection alpha value
        self._alpha = None
        self.set_facecolors(zalpha(self._facecolor3d, vzs))
        self.set_edgecolors(zalpha(self._edgecolor3d, vzs))
        PatchCollection.set_offsets(self, zip(vxs, vys))

        return min(vzs)

    def draw(self, renderer):
        PatchCollection.draw(self, renderer)

def patch_collection_2d_to_3d(col, zs=0, dir='z'):
    """Convert a PatchCollection to a Patch3DCollection object."""
    col.__class__ = Patch3DCollection
    col.set_3d_properties(zs, dir)

class Poly3DCollection(PolyCollection):

    def __init__(self, verts, *args, **kwargs):
        PolyCollection.__init__(self, verts, *args, **kwargs)
        self.set_3d_properties()

    def get_vector(self, segments3d):
        """optimise points for projection"""
        si = 0
        ei = 0
        segis = []
        points = []
        for p in segments3d:
            points.extend(p)
            ei = si+len(p)
            segis.append((si,ei))
            si = ei
        xs,ys,zs = zip(*points)
        ones = np.ones(len(xs))
        self._vec = np.array([xs,ys,zs,ones])
        self._segis = segis
        self._sort_zpos = min(zs)

    def set_verts(self, verts, closed=True):
        self.get_vector(verts)
        # 2D verts will be updated at draw time
        PolyCollection.set_verts(self, [], closed)

    def set_3d_properties(self):
        self._zsort = 1
        self._facecolors3d = PolyCollection.get_facecolors(self)
        self._edgecolors3d = self.get_edgecolors()

    def do_3d_projection(self, renderer):
        txs, tys, tzs = proj3d.proj_transform_vec(self._vec, renderer.M)
        xyzlist = [(txs[si:ei], tys[si:ei], tzs[si:ei]) \
                for si, ei in self._segis]
        colors = self._facecolors3d

        # if required sort by depth (furthest drawn first)
        if self._zsort:
            z_segments_2d = [(min(zs),zip(xs,ys),c) for
                             (xs,ys,zs),c in zip(xyzlist,colors)]
            z_segments_2d.sort()
            z_segments_2d.reverse()
        else:
            raise ValueError, "whoops"
        segments_2d = [s for z,s,c in z_segments_2d]
        colors = [c for z,s,c in z_segments_2d]
        PolyCollection.set_verts(self, segments_2d)
        self._facecolors2d = colors

        # Return zorder value
        zvec = np.array([[0], [0], [self._sort_zpos], [1]])
        ztrans = proj3d.proj_transform_vec(zvec, renderer.M)
        return ztrans[2][0]

    def get_facecolors(self):
        return self._facecolors2d
    get_facecolor = get_facecolors

    def draw(self, renderer):
        return Collection.draw(self, renderer)

def poly_collection_2d_to_3d(col, zs=None, dir='z'):
    """Convert a PolyCollection to a Poly3DCollection object."""
    segments_3d = paths_to_3d_segments(col.get_paths(), zs, dir)
    col.__class__ = Poly3DCollection
    col.set_verts(segments_3d)
    col.set_3d_properties()

def juggle_axes(xs,ys,zs, dir):
    """
    Depending on the direction of the plot re-order the axis.
    This is so that 2d plots can be plotted along any direction.
    """
    if dir == 'x': return zs,xs,ys
    elif dir == 'y': return xs,zs,ys
    else: return xs,ys,zs

def iscolor(c):
    try:
        return (len(c) == 4 or len(c) == 3) and hasattr(c[0], '__float__')
    except (IndexError):
        return False

def get_colors(c, num):
    """Stretch the color argument to provide the required number num"""

    if type(c)==type("string"):
        c = colors.colorConverter.to_rgba(colors)

    if iscolor(c):
        return [c] * num
    if len(c) == num:
        return c
    elif iscolor(c):
        return [c] * num
    elif iscolor(c[0]):
        return [c[0]] * num
    else:
        raise ValueError, 'unknown color format %s' % c

def zalpha(colors, zs):
    """Modify the alphas of the color list according to depth"""
    colors = get_colors(colors,len(zs))
    norm = Normalize(min(zs),max(zs))
    sats = 1 - norm(zs)*0.7
    colors = [(c[0],c[1],c[2],c[3]*s) for c,s in zip(colors,sats)]
    return colors

