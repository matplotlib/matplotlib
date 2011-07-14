"""
This module contains functions to handle markers.  Used by both the
marker functionality of `~matplotlib.axes.Axes.plot` and
`~matplotlib.axes.Axes.scatter`.
"""

import numpy as np

from cbook import is_math_text
from path import Path
from transforms import IdentityTransform, Affine2D

# special-purpose marker identifiers:
(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN,
 CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN) = range(8)

# TODO: Cache the marker path within the object

class MarkerStyle:
    style_table = """
======================== =====================================================
marker                   description
======================== =====================================================
%s
``'$...$'``              render the string using mathtext
(numsides, style, angle) where style is 1: star, 2: asterisk, 3: circle
(verts, 0)               where verts is a list of (x, y) pairs in range (0, 1)
======================== =====================================================
""" 
    
    # TODO: Automatically generate this
    accepts = """ACCEPTS: [ %s | ``'$...$'`` | tuple ]"""
    
    markers =  {
        '.'        : 'point',
        ','        : 'pixel',
        'o'        : 'circle',
        'v'        : 'triangle_down',
        '^'        : 'triangle_up',
        '<'        : 'triangle_left',
        '>'        : 'triangle_right',
        '1'        : 'tri_down',
        '2'        : 'tri_up',
        '3'        : 'tri_left',
        '4'        : 'tri_right',
        's'        : 'square',
        'p'        : 'pentagon',
        '*'        : 'star',
        'h'        : 'hexagon1',
        'H'        : 'hexagon2',
        '+'        : 'plus',
        'x'        : 'x',
        'D'        : 'diamond',
        'd'        : 'thin_diamond',
        '|'        : 'vline',
        '_'        : 'hline',
        TICKLEFT   : 'tickleft',
        TICKRIGHT  : 'tickright',
        TICKUP     : 'tickup',
        TICKDOWN   : 'tickdown',
        CARETLEFT  : 'caretleft',
        CARETRIGHT : 'caretright',
        CARETUP    : 'caretup',
        CARETDOWN  : 'caretdown',
        None       : 'nothing',
        ' '        : 'nothing',
        ''         : 'nothing'
    }

    filled_markers = ('o', '^', 'v', '<', '>',
                      's', 'd', 'D', 'h', 'H', 'p', '*')

    fillstyles = ('full', 'left' , 'right' , 'bottom' , 'top')

    # TODO: Is this ever used as a non-constant?
    _point_size_reduction = 0.5
    
    def __init__(self, marker=None, fillstyle='full'):
        self._fillstyle = fillstyle
        self.set_marker(marker)
        self.set_fillstyle(fillstyle)

    def _recache(self):
        (self._path, self._transform,
         self._alt_path, self._alt_transform,
         self._snap_threshold) = self._marker_function()
        
    def __nonzero__(self):
        return len(self._path.vertices)
        
    def is_filled(self):
        return (self._marker in self.filled_markers
                or is_math_text(self._marker))

    def get_fillstyle(self):
        return self._fillstyle

    def set_fillstyle(self, fillstyle):
        # TODO: Raise exception for markers where fillstyle doesn't make sense
        assert fillstyle in self.fillstyles
        self._fillstyle = fillstyle
        self._recache()
        
    def get_marker(self):
        return self._marker

    def set_marker(self, marker):
        if marker in self.markers:
            self._marker = marker
            self._marker_function = getattr(
                self, '_get_' + self.markers[marker])
        elif is_math_text(marker):
            self._marker = marker
            self._marker_function = self._get_mathtext_path
        else:
            raise ValueError('Unrecognized marker style %s' % marker)

        self._recache()

    def get_path(self):
        return self._path

    def get_transform(self):
        return self._transform.frozen()

    def get_alt_path(self):
        return self._alt_path

    def get_alt_transform(self):
        return self._alt_transform.frozen()

    def get_snap_threshold(self):
        return self._snap_threshold
        
    def _get_nothing(self):
        return Path(np.empty((0,2))), IdentityTransform(), None, None, False
        
    def _get_mathtext_path(self):
        """
        Draws mathtext markers '$...$' using TextPath object.

        Submitted by tcb
        """
        from matplotlib.patches import PathPatch
        from matplotlib.text import TextPath

        # again, the properties could be initialised just once outside
        # this function
        # Font size is irrelevant here, it will be rescaled based on
        # the drawn size later
        props = FontProperties(size=1.0)
        text = TextPath(xy=(0,0), s=self.get_marker(), fontproperties=props,
                        usetex=rcParams['text.usetex'])
        if len(text.vertices) == 0:
            return text, IdentityTransform(), False
        
        xmin, ymin = text.vertices.min(axis=0)
        xmax, ymax = text.vertices.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin
        max_dim = max(width, height)
        path_trans = Affine2D() \
            .translate(-xmin + 0.5 * -width, -ymin + 0.5 * -height) \
            .scale(1.0 / max_dim)

        return text, path_trans, None, None, False

    def _get_circle(self, reduction = 1.0):
        transform = Affine2D().scale(0.5 * reduction)
        fs = self.get_fillstyle()
        if fs=='full':
            return Path.unit_circle(), transform, None, None, 3.0
        else:
            # build a right-half circle
            if fs=='bottom': rotate = 270.
            elif fs=='top': rotate = 90.
            elif fs=='left': rotate = 180.
            else: rotate = 0.

            half = Path.unit_circle_righthalf()
            transform = transform.rotate_deg(rotate)
            alt_transform = transform.rotate_deg(180.)
            return half, transform, half, alt_transform, 3.0

    def _get_pixel(self):
        return Path.unit_rectangle(), Affine2D().translate(-0.5, 0.5), None, None, False

    def _get_point(self):
        return self._get_circle(reduction = self._point_size_reduction)

    _triangle_path = Path(
        [[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # Going down halfway looks to small.  Golden ratio is too far.
    _triangle_path_u = Path(
        [[0.0, 1.0], [-3/5., -1/5.], [3/5., -1/5.], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_d = Path(
        [[-3/5., -1/5.], [3/5., -1/5.], [1.0, -1.0], [-1.0, -1.0], [-3/5., -1/5.]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_l = Path(
        [[0.0, 1.0], [0.0, -1.0], [-1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_r = Path(
        [[0.0, 1.0], [0.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    def _get_triangle(self, rot, skip):
        direction_map = {
            'up': (0.0, 0),
            'down': (180.0, 2),
            'left': (90.0, 3),
            'right': (270.0, 1)
            }
        rot, skip = direction_map[direction]
        transform = Affine2D().scale(0.5, 0.5).rotate_deg(rot)
        fs = self.get_fillstyle()

        if fs=='full':
            return self._triangle_path, transform, None, None, 5.0
        else:
            rgbFace_alt = self._get_rgb_face(alt=True)

            mpaths = [self._triangle_path_u,
                      self._triangle_path_l,
                      self._triangle_path_d,
                      self._triangle_path_r]
            
            if fs=='top':
                mpath     = mpaths[(0+skip) % 4]
                mpath_alt = mpaths[(2+skip) % 4]
            elif fs=='bottom':
                mpath     = mpaths[(2+skip) % 4]
                mpath_alt = mpaths[(0+skip) % 4]
            elif fs=='left':
                mpath     = mpaths[(1+skip) % 4]
                mpath_alt = mpaths[(3+skip) % 4]
            else:
                mpath     = mpaths[(3+skip) % 4]
                mpath_alt = mpaths[(1+skip) % 4]

            return mpath, transform, mpath_alt, transform, 5.0

    def _get_triangle_up(self):
        self._get_triangle(0.0, 0)

    def _get_triangle_down(self):
        self._get_triangle(180.0, 2)

    def _get_triangle_left(self):
        self._get_triangle(90.0, 3)

    def _get_triangle_right(self):
        self._get_triangle(270.0, 1)

    def _get_square(self):
        transform = Affine2D().translate(-0.5, -0.5)
        fs = self.get_fillstyle()
        if fs=='full':
            return Path.unit_rectangle(), transform, None, None, 2.0
        else:
            # build a bottom filled square out of two rectangles, one
            # filled.  Use the rotation to support left, right, bottom
            # or top
            if fs=='bottom': rotate = 0.
            elif fs=='top': rotate = 180.
            elif fs=='left': rotate = 270.
            else: rotate = 90.

            bottom = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]])
            top = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
            transform = transform.rotate_deg(rotate)
            return bottom, transform, top, transform, 2.0

    def _get_diamond(self):
        transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
        fs = self.get_fillstyle()
        if fs=='full':
            return Path.unit_rectangle(), transform, None, None, 5.0
        else:
            right = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
            left = Path([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

            if fs=='bottom': rotate = 270.
            elif fs=='top': rotate = 90.
            elif fs=='left': rotate = 180.
            else: rotate = 0.

            transform = transform.rotate_deg(rotate)

            return right, transform, left, transform, 5.0

    def _get_thin_diamond(self):
        right, transform, left, _, snap = self._get_diamond()
        transform = transform.scale(0.6, 1.0)
        return right, transform, left, transform, 3.0

    def _get_pentagon(self):
        transform = Affine2D().scale(0.5)
        polypath = Path.unit_regular_polygon(5)

        fs = self.get_fillstyle()

        if fs == 'full':
            return polypath, tranform, None, None, 5.0
        else:
            verts = polypath.vertices

            y = (1+np.sqrt(5))/4.
            top = Path([verts[0], verts[1], verts[4], verts[0]])
            bottom = Path([verts[1], verts[2], verts[3], verts[4], verts[1]])
            left = Path([verts[0], verts[1], verts[2], [0,-y], verts[0]])
            right = Path([verts[0], verts[4], verts[3], [0,-y], verts[0]])

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            return mpath, transform, mpath_alt, transform, 5.0

    def _get_star(self):
        transform = Affine2D().scale(0.5)
        fs = self.get_fillstyle()

        polypath = Path.unit_regular_star(5, innerCircle=0.381966)

        if fs == 'full':
            return polypath, transform, None, None, 5.0
        else:
            verts = polypath.vertices

            top = Path(np.vstack((verts[0:4,:], verts[7:10,:], verts[0])))
            bottom = Path(np.vstack((verts[3:8,:], verts[3])))
            left = Path(np.vstack((verts[0:6,:], verts[0])))
            right = Path(np.vstack((verts[0], verts[5:10,:], verts[0])))

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            return mpath, transform, mpath_alt, transform, 5.0

    def _get_hexagon1(self):
        transform = Affine2D().scale(0.5)
        fs = self.get_fillstyle()

        polypath = Path.unit_regular_polygon(6)

        if fs == 'full':
            return polypath, transform, None, None, 5.0
        else:
            verts = polypath.vertices

            # not drawing inside lines
            x = np.abs(np.cos(5*np.pi/6.))
            top = Path(np.vstack(([-x,0],verts[(1,0,5),:],[x,0])))
            bottom = Path(np.vstack(([-x,0],verts[2:5,:],[x,0])))
            left = Path(verts[(0,1,2,3),:])
            right = Path(verts[(0,5,4,3),:])

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            return mpath, transform, mpath_alt, transform, 5.0

    def _get_hexagon2(self):
        transform = Affine2D().scale(0.5).rotate_deg(30)
        fs = self.get_fillstyle()

        polypath = Path.unit_regular_polygon(6)

        if fs == 'full':
            return polypath, transform, None, None, 5.0
        else:
            verts = polypath.vertices

            # not drawing inside lines
            x, y = np.sqrt(3)/4, 3/4.
            top = Path(verts[(1,0,5,4,1),:])
            bottom = Path(verts[(1,2,3,4),:])
            left = Path(np.vstack(([x,y],verts[(0,1,2),:],[-x,-y],[x,y])))
            right = Path(np.vstack(([x,y],verts[(5,4,3),:],[-x,-y])))

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            return mpath, transform, mpath_alt, transform, 5.0

    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])
    def _get_vline(self):
        transform = Affine2D().scale(0.5)
        return self._line_marker_path, transform, None, None, 1.0

    def _get_hline(self):
        transform = Affine2D().scale(0.5).rotate_deg(90)
        return self._line_marker_path, transform, None, None, 1.0

    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])
    def _get_tickleft(self):
        transform = Affine2D().scale(-1.0, 1.0)
        return self._tickhoriz_path, transform, None, None, 1.0

    def _get_tickright(self):
        transform = Affine2D().scale(1.0, 1.0)
        return self._tickhoriz_path, transform, None, None, 1.0
        
    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])
    def _get_tickup(self):
        transform = Affine2D().scale(1.0, 1.0)
        return self._tickvert_path, transform, None, None, 1.0
        
    def _get_tickdown(self):
        transform = Affine2D().scale(1.0, -1.0)
        return self._tickvert_path, transform, None, None, 1.0

    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0],
                       [0.0, -1.0], [0.0, 1.0]],
                      [Path.MOVETO, Path.LINETO,
                       Path.MOVETO, Path.LINETO])
    def _get_plus(self):
        transform = Affine2D().scale(0.5)
        return self._plus_path, transform, None, None, 1.0

    _tri_path = Path([[0.0, 0.0], [0.0, -1.0],
                      [0.0, 0.0], [0.8, 0.5],
                      [0.0, 0.0], [-0.8, 0.5]],
                     [Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO])
    def _draw_tri_down(self):
        transform = Affine2D().scale(0.5)
        return self._tri_path, transform, None, None, 5.0

    def _draw_tri_up(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(90)
        return self._tri_path, transform, None, None, 5.0

    def _draw_tri_left(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(270)
        return self._tri_path, transform, None, None, 5.0

    def _draw_tri_right(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(180)

    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])
    def _draw_caretdown(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5)
        return self._caret_path, transform, None, None, 3.0

    def _draw_caretup(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(180)
        return self._caret_path, transform, None, None, 3.0

    def _draw_caretleft(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(270)
        return self._caret_path, transform, None, None, 3.0

    def _draw_caretright(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5).rotate_deg(90)
        return self._caret_path, transform, None, None, 3.0

    _x_path = Path([[-1.0, -1.0], [1.0, 1.0],
                    [-1.0, 1.0], [1.0, -1.0]],
                   [Path.MOVETO, Path.LINETO,
                    Path.MOVETO, Path.LINETO])
    def _draw_x(self, renderer, gc, path, path_trans):
        transform = Affine2D().scale(0.5)
        return self._x_path, transform, None, None, 3.0

_styles = [(repr(x), y) for x, y in MarkerStyle.markers.items()]
_styles.sort()
MarkerStyle.style_table = (
    MarkerStyle.style_table %
    '\n'.join(['``%7s`` %33s' % (x, y) for (x, y) in _styles]))

MarkerStyle.accepts = (
    MarkerStyle.accepts %
    ' | '.join(['``%s``' % x for (x, y) in _styles]))
