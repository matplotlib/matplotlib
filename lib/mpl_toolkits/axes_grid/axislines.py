"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible to draw axis line,
ticks, ticklabel and axis labels are separated out from the mpl's Axis
class, which are much more than artists in the original
mpl. Originally, this change was motivated to support curvlinear
grid. Here are a few reasons that I came up with new axes class.


 * "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
   different ticks (tick locations and labels). This is not possible
   with the current mpl, although some twin axes trick can help.

 * Curvelinear grid.

 * angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and
has following children artists which will draw ticks, labels, etc.

 * line
 * major_ticks, major_ticklabels
 * minor_ticks, minor_ticklabels
 * offsetText
 * label

Note that these are separate artists from Axis class of the
original mpl, thus most of tick-related command in the original mpl
won't work, although some effort has made to work with. For example,
color and markerwidth of the ax.axis["bottom"].major_ticks will follow
those of Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.

"""

import matplotlib.axes as maxes
import matplotlib.artist as martist
import matplotlib.text as mtext
import matplotlib.font_manager as font_manager

from matplotlib.path import Path
from matplotlib.transforms import Affine2D, ScaledTranslation, \
     IdentityTransform, TransformedPath, Bbox
from matplotlib.collections import LineCollection

from matplotlib import rcParams
import warnings

import numpy as np


import matplotlib.lines as mlines



class BezierPath(mlines.Line2D):

    def __init__(self, path, *kl, **kw):
        mlines.Line2D.__init__(self, [], [], *kl, **kw)
        self._path = path
        self._invalid = False

    def recache(self):

        self._transformed_path = TransformedPath(self._path, self.get_transform())

        self._invalid = False

    def set_path(self, path):
        self._path = path
        self._invalid = True


    def draw(self, renderer):
        if self._invalid:
            self.recache()

        renderer.open_group('line2d')

        if not self._visible: return
        gc = renderer.new_gc()
        self._set_gc_clip(gc)

        gc.set_foreground(self._color)
        gc.set_antialiased(self._antialiased)
        gc.set_linewidth(self._linewidth)
        gc.set_alpha(self._alpha)
        if self.is_dashed():
            cap = self._dashcapstyle
            join = self._dashjoinstyle
        else:
            cap = self._solidcapstyle
            join = self._solidjoinstyle
        gc.set_joinstyle(join)
        gc.set_capstyle(cap)

        funcname = self._lineStyles.get(self._linestyle, '_draw_nothing')
        if funcname != '_draw_nothing':
            tpath, affine = self._transformed_path.get_transformed_path_and_affine()
            lineFunc = getattr(self, funcname)
            lineFunc(renderer, gc, tpath, affine.frozen())

        gc.restore()
        renderer.close_group('line2d')



class UnimplementedException(Exception):
    pass



class AxisArtistHelper(object):
    """
    AxisArtistHelper should define
    following method with given APIs. Note that the first axes argument
    will be axes attribute of the caller artist.


        # LINE

        def get_line(self, axes):
            # path : Path
            return path

        def get_line_transform(self, axes):
            # ...
            # trans : transform
            return trans

        # LABEL

        def get_label_pos(self, axes):
            # x, y : position
            return (x, y), trans


        def get_label_offset_transform(self, \
                axes,
                pad_points, fontprops, renderer,
                bboxes,
                ):
            # va : vertical alignment
            # ha : horizontal alignment
            # a : angle
            return trans, va, ha, a

        # TICK

        def get_tick_transform(self, axes):
            return trans

        def get_tick_iterators(self, axes):
            # iter : iteratoable object that yields (c, angle, l) where
            # c, angle, l is position, tick angle, and label

            return iter_major, iter_minot


        """
    class _Base(object):

        def __init__(self, label_direction):
            self.label_direction = label_direction

        #def update(self):
        #    raise UnimplementedException("update method not implemented")

        def update_lim(self, axes):
            pass

        _label_angles = dict(left=90, right=90, bottom=0, top=0)
        _ticklabel_angles = dict(left=0, right=0, bottom=0, top=0)

        def _get_label_offset_transform(self, pad_points, fontprops, renderer,
                                        bboxes=None,
                                        #trans=None
                                        ):

            """
            Returns (offset-transform, vertical-alignment, horiz-alignment)
            of (tick or axis) labels appropriate for the label
            direction.

            The offset-transform represents a required pixel offset
            from the reference point. For example, x-axis center will
            be the referece point for xlabel.

            pad_points : padding from axis line or tick labels (see bboxes)
            fontprops : font properties for label
            renderer : renderer
            bboxes=None : list of bboxes (window extents) of the tick labels.

            all the above parameters are used to estimate the offset.

            """
            if renderer:
                pad_pixels = renderer.points_to_pixels(pad_points)
                font_size_points = fontprops.get_size_in_points()
                font_size_pixels = renderer.points_to_pixels(font_size_points)
            else:
                pad_pixels = pad_points
                font_size_points = fontprops.get_size_in_points()
                font_size_pixels = font_size_points

            if bboxes:
                bbox = Bbox.union(bboxes)
                w, h = bbox.width, bbox.height
            else:
                w, h = 0, 0


            tr = Affine2D()
            if self.label_direction == "left":
                tr.translate(-(pad_pixels+w), 0.)
                #trans = trans + tr

                return tr, "center", "right"

            elif self.label_direction == "right":
                tr.translate(+(pad_pixels+w), 0.)
                #trans = trans + tr

                return tr, "center", "left"

            elif self.label_direction == "bottom":
                tr.translate(0, -(pad_pixels+font_size_pixels+h))
                #trans = trans + tr

                return tr, "baseline", "center"

            elif self.label_direction == "top":
                tr.translate(0, +(pad_pixels+h))
                #trans = trans + tr

                return tr, "baseline", "center"

            else:
                raise ValueError("")


        def get_label_offset_transform(self,
                                       axes,
                                       pad_points, fontprops, renderer,
                                       bboxes,
                                       #trans=None
                                       ):

            tr, va, ha = self._get_label_offset_transform(pad_points, fontprops,
                                                          renderer,
                                                          bboxes,
                                                          #trans
                                                          )

            a = self._label_angles[self.label_direction]
            return tr, va, ha, a


        def get_ticklabel_offset_transform(self, axes,
                                           pad_points, fontprops,
                                           renderer,
                                           ):

            tr, va, ha = self._get_label_offset_transform(pad_points, fontprops,
                                                          renderer,
                                                          None,
                                                          )

            a = self._ticklabel_angles[self.label_direction]
            return tr, va, ha, a



    class Fixed(_Base):

        _default_passthru_pt = dict(left=(0, 0),
                                    right=(1, 0),
                                    bottom=(0, 0),
                                    top=(0, 1))

        def __init__(self,
                     loc, nth_coord=None,
                     passingthrough_point=None, label_direction=None):
            """
            nth_coord = along which coordinate value varies
            in 2d, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
            """
            if loc not in ["left", "right", "bottom", "top"]:
                raise ValueError("%s" % loc)

            if nth_coord is None:
                if loc in ["left", "right"]:
                    nth_coord = 1
                elif loc in ["bottom", "top"]:
                    nth_coord = 0

            self.nth_coord = nth_coord

            super(AxisArtistHelper.Fixed, self).__init__(loc)

            if passingthrough_point is None:
                passingthrough_point = self._default_passthru_pt[loc]
            if label_direction is None:
                label_direction = loc


            self.passthru_pt = passingthrough_point

            _verts = np.array([[0., 0.],
                               [1., 1.]])
            fixed_coord = 1-nth_coord
            _verts[:,fixed_coord] = self.passthru_pt[fixed_coord]

            # axis line in transAxes
            self._path = Path(_verts)


        def get_nth_coord(self):
            return self.nth_coord

        # LINE

        def get_line(self, axes):
            return self._path

        def get_line_transform(self, axes):
            return axes.transAxes

        # LABLE

        def get_label_pos(self, axes):
            """
            label reference position in transAxes.

            get_label_transform() returns a transform of (transAxes+offset)
            """
            _verts = [0.5, 0.5]
            nth_coord = self.nth_coord
            fixed_coord = 1-nth_coord
            _verts[fixed_coord] = self.passthru_pt[fixed_coord]
            return _verts, axes.transAxes


        def get_label_offset_transform(self, axes,
                                       pad_points, fontprops, renderer,
                                       bboxes,
                                       ):

            tr, va, ha = self._get_label_offset_transform( \
                pad_points, fontprops, renderer, bboxes,
                #trans
                )

            a = self._label_angles[self.label_direction]
            #tr = axes.transAxes + tr

            return tr, va, ha, a



        # TICK

        def get_tick_transform(self, axes):
            trans_tick = [axes.get_xaxis_transform(),
                          axes.get_yaxis_transform()][self.nth_coord]

            return trans_tick



    class Floating(_Base):
        def __init__(self, nth_coord,
                     passingthrough_point, label_direction, transform):

            self.nth_coord = nth_coord

            self.passingthrough_point = passingthrough_point

            self.transform = transform

            super(AxisArtistHelper.Floating,
                  self).__init__(label_direction)


        def get_nth_coord(self):
            return self.nth_coord

        def get_line(self, axes):
            _verts = np.array([[0., 0.],
                               [1., 1.]])

            fixed_coord = 1-self.nth_coord
            trans_passingthrough_point = self.transform + axes.transAxes.inverted()
            p = trans_passingthrough_point.transform_point(self.passingthrough_point)
            _verts[:,fixed_coord] = p[fixed_coord]

            return Path(_verts)

        def get_line_transform(self, axes):
            return axes.transAxes

        def get_label_pos(self, axes):
            _verts = [0.5, 0.5]

            fixed_coord = 1-self.nth_coord
            trans_passingthrough_point = self.transform + axes.transAxes.inverted()
            p = trans_passingthrough_point.transform_point(self.passingthrough_point)
            _verts[fixed_coord] = p[fixed_coord]
            if not (0. <= _verts[fixed_coord] <= 1.):
                return None, None
            else:
                return _verts, axes.transAxes

        def get_label_transform(self, axes,
                                pad_points, fontprops, renderer,
                                bboxes,
                                ):

            tr, va, ha = self._get_label_offset_transform(pad_points, fontprops,
                                                          renderer,
                                                          bboxes,
                                                          #trans
                                                          )

            a = self._label_angles[self.label_direction]
            tr = axes.transAxes + tr
            #tr = axes.transAxes + tr

            return tr, va, ha, a



        def get_tick_transform(self, axes):
            return self.transform





class AxisArtistHelperRectlinear:

    class Fixed(AxisArtistHelper.Fixed):

        def __init__(self,
                     axes, loc, nth_coord=None,
                     passingthrough_point=None, label_direction=None):
            """
            nth_coord = along which coordinate value varies
            in 2d, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
            """

            super(AxisArtistHelperRectlinear.Fixed, self).__init__( \
                     loc, nth_coord,
                     passingthrough_point, label_direction)

            self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]



        # TICK

        def get_tick_iterators(self, axes):
            """tick_loc, tick_angle, tick_label"""

            angle = 0 - 90 * self.nth_coord
            if self.passthru_pt[1 - self.nth_coord] > 0.5:
                angle = 180+angle

            major = self.axis.major
            majorLocs = major.locator()
            major.formatter.set_locs(majorLocs)
            majorLabels = [major.formatter(val, i) for i, val in enumerate(majorLocs)]

            minor = self.axis.minor
            minorLocs = minor.locator()
            minor.formatter.set_locs(minorLocs)
            minorLabels = [minor.formatter(val, i) for i, val in enumerate(minorLocs)]

            trans_tick = self.get_tick_transform(axes)

            tr2ax = trans_tick + axes.transAxes.inverted()

            def _f(locs, labels):
                for x, l in zip(locs, labels):

                    c = list(self.passthru_pt) # copy
                    c[self.nth_coord] = x

                    # check if the tick point is inside axes
                    c2 = tr2ax.transform_point(c)
                    delta=0.001
                    if 0. -delta<= c2[self.nth_coord] <= 1.+delta:
                        yield c, angle, l

            return _f(majorLocs, majorLabels), _f(minorLocs, minorLabels)



    class Floating(AxisArtistHelper.Floating):
        def __init__(self, axes, nth_coord,
                     passingthrough_point, label_direction, transform):

            super(AxisArtistHelperRectlinear.Floating, self).__init__( \
                nth_coord, passingthrough_point, label_direction, transform)

            self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]


        def get_tick_iterators(self, axes):
            """tick_loc, tick_angle, tick_label"""

            angle = 0 - 90 * self.nth_coord

            major = self.axis.major
            majorLocs = major.locator()
            major.formatter.set_locs(majorLocs)
            majorLabels = [major.formatter(val, i) for i, val in enumerate(majorLocs)]

            minor = self.axis.minor
            minorLocs = minor.locator()
            minor.formatter.set_locs(minorLocs)
            minorLabels = [minor.formatter(val, i) for i, val in enumerate(minorLocs)]

            tr2ax = self.transform + axes.transAxes.inverted()

            def _f(locs, labels):
                for x, l in zip(locs, labels):

                    c = list(self.passingthrough_point) # copy
                    c[self.nth_coord] = x
                    c1, c2 = tr2ax.transform_point(c)
                    if 0. <= c1 <= 1. and 0. <= c2 <= 1.:
                        yield c, angle, l

            return _f(majorLocs, majorLabels), _f(minorLocs, minorLabels)





class GridHelperBase(object):

    def __init__(self):
        self._force_update = True
        self._old_limits = None
        super(GridHelperBase, self).__init__()


    def update_lim(self, axes):
        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()

        if self._force_update or self._old_limits != (x1, x2, y1, y2):
            self._update(x1, x2, y1, y2)
            self._force_update = False
            self._old_limits = (x1, x2, y1, y2)


    def _update(self, x1, x2, y1, y2):
        pass


    def invalidate(self):
        self._force_update = True


    def get_gridlines(self):
        return []



class GridHelperRectlinear(GridHelperBase):


    def __init__(self, axes):

        super(GridHelperRectlinear, self).__init__()
        self.axes = axes

    #def set_axes(self, axes):
    #    self.axes = axes

    def _get_axisline_helper_deprecated(self, nth_coord, loc,
                             passingthrough_point, transform=None):
        if transform is None or transform is self.axes.transAxes:
            return AxisArtistHelper.Fixed(self.axes, loc,
                                        nth_coord, passingthrough_point)

        else:
            label_direction = loc
            return AxisArtistHelper.Floating(self.axes,
                                           nth_coord, passingthrough_point,
                                           label_direction,
                                           transform)


    def new_fixed_axis(self, loc,
                       nth_coord=None, passthrough_point=None,
                       #transform=None,
                       tick_direction="in",
                       label_direction=None,
                       offset=None,
                       axes=None,
                       ):

        if axes is None:
            warnings.warn("'new_fixed_axis' explicitly requires the axes keyword.")
            axes = self.axes

        _helper = AxisArtistHelperRectlinear.Fixed(axes, loc,
                                                   nth_coord,
                                                   passthrough_point)

        axisline = AxisArtist(axes, _helper,
                              #tick_direction="in",
                              offset=offset,
                              )

        return axisline


    def new_floating_axis(self, nth_coord=None, passthrough_point=None,
                          transform=None,
                          tick_direction="in",
                          label_direction=None,
                       axes=None,
                       ):

        if axes is None:
            warnings.warn("'new_floating_axis' explicitly requires the axes keyword.")
            axes = self.axes

        _helper = AxisArtistHelperRectlinear.Floating( \
            axes,
            nth_coord, passthrough_point,
            label_direction,
            transform)

        axisline = AxisArtist(axes, _helper,
                              #tick_direction="in",
                              )

        return axisline


    def new_axisline_deprecated(self, loc,
                     nth_coord=None, passthrough_point=None,
                     transform=None,
                     tick_direction="in",
                     label_direction=None,
                     offset=None):

        warnings.warn("new_axisline is deprecated. Use new_fixed_axis "
                      "or new_floating_axis instead")

        _helper = self._get_axisline_helper(nth_coord, loc,
                                            passthrough_point,
                                            transform)

        axisline = AxisArtist(self.axes, _helper,
                            #tick_direction="in",
                            offset=offset,
                            )

        return axisline



from matplotlib.lines import Line2D

class Ticks(Line2D):
    def __init__(self, ticksize, **kwargs):
        self.ticksize = ticksize
        self.locs_angles = []

        self._axis = kwargs.pop("axis", None)
        if self._axis is not None:
            if "color" not in kwargs:
                kwargs["color"] = "auto"
            if ("mew" not in kwargs) and ("markeredgewidth" not in kwargs):
                kwargs["markeredgewidth"] = "auto"

        super(Ticks, self).__init__([0.], [0.], **kwargs)


    def get_color(self):
        if self._color == 'auto':
            if self._axis is not None:
                ticklines = self._axis.get_ticklines()
                if ticklines:
                    color_from_axis = ticklines[0].get_color()
                    return color_from_axis
            return "k"

        return super(Ticks, self).get_color()


    def get_markeredgecolor(self):
        if self._markeredgecolor == 'auto':
            return self.get_color()
        else:
            return self._markeredgecolor

    def get_markeredgewidth(self):
        if self._markeredgewidth == 'auto':
            if self._axis is not None:
                ticklines = self._axis.get_ticklines()
                if ticklines:
                    width_from_axis = ticklines[0].get_markeredgewidth()
                    return width_from_axis
            return .5

        else:
            return self._markeredgewidth


    def update_locs_angles(self, locs_angles, renderer):
        self.locs_angles = locs_angles

    _tickvert_path = Path([[0., 0.], [0., 1.]])

    def draw(self, renderer):
        if not self.get_visible():
            return
        size = self.ticksize
        path_trans = self.get_transform()

        # set gc : copied from lines.py
#         gc = renderer.new_gc()
#         self._set_gc_clip(gc)

#         gc.set_foreground(self.get_color())
#         gc.set_antialiased(self._antialiased)
#         gc.set_linewidth(self._linewidth)
#         gc.set_alpha(self._alpha)
#         if self.is_dashed():
#             cap = self._dashcapstyle
#             join = self._dashjoinstyle
#         else:
#             cap = self._solidcapstyle
#             join = self._solidjoinstyle
#         gc.set_joinstyle(join)
#         gc.set_capstyle(cap)
#         gc.set_snap(self.get_snap())


        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_foreground(self.get_markeredgecolor())
        gc.set_linewidth(self.get_markeredgewidth())
        gc.set_alpha(self._alpha)

        offset = renderer.points_to_pixels(size)
        marker_scale = Affine2D().scale(offset, offset)

        for loc, angle in self.locs_angles:

            marker_rotation = Affine2D().rotate_deg(angle)
            #marker_rotation.clear().rotate_deg(angle)
            marker_transform = marker_scale + marker_rotation
            locs = path_trans.transform_non_affine(np.array([loc]))
            renderer.draw_markers(gc, self._tickvert_path, marker_transform,
                                  Path(locs), path_trans.get_affine())

        gc.restore()




class TickLabels(mtext.Text):

    def __init__(self, size, **kwargs):
        self._locs_labels = []

        self._axis = kwargs.pop("axis", None)
        if self._axis is not None:
            if "color" not in kwargs:
                kwargs["color"] = "auto"

        super(TickLabels, self).__init__(x=0., y=0., text="",
                                         **kwargs
                                         )

    def update_locs_labels(self, locs_labels, renderer):
        self._locs_labels = locs_labels

    def get_color(self):
        if self._color == 'auto':
            if self._axis is not None:
                ticklabels = self._axis.get_ticklabels()
                if ticklabels:
                    color_from_axis = ticklabels[0].get_color()
                    return color_from_axis
            return "k"

        return super(TickLabels, self).get_color()


    def draw(self, renderer):
        if not self.get_visible(): return

        for (x, y), l in self._locs_labels:
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            super(TickLabels, self).draw(renderer)

    def get_window_extents(self, renderer):
        bboxes = []
        for (x, y), l in self._locs_labels:
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)

            bboxes.append(self.get_window_extent())

        return [b for b in bboxes if b.width!=0 or b.height!=0]
        #if bboxes:
        #    return Bbox.union([b for b in bboxes if b.width!=0 or b.height!=0])
        #else:
        #    return Bbox.from_bounds(0, 0, 0, 0)





class AxisLabel(mtext.Text):
    def __init__(self, *kl, **kwargs):
        self._axis = kwargs.pop("axis", None)
        if self._axis is not None:
            if "color" not in kwargs:
                kwargs["color"] = "auto"

        super(AxisLabel, self).__init__(*kl, **kwargs)

    def get_color(self):
        if self._color == 'auto':
            if self._axis is not None:
                label = self._axis.get_label()
                if label:
                    color_from_axis = label.get_color()
                    return color_from_axis
            return "k"

        return super(AxisLabel, self).get_color()

    def get_text(self):
        t = super(AxisLabel, self).get_text()
        if t == "__from_axes__":
            return self._axis.get_label().get_text()
        return self._text


class GridlinesCollection(LineCollection):
    def __init__(self, *kl, **kwargs):
        super(GridlinesCollection, self).__init__(*kl, **kwargs)
        self.set_grid_helper(None)

    def set_grid_helper(self, grid_helper):
        self._grid_helper = grid_helper

    def draw(self, renderer):
        if self._grid_helper is not None:
            self._grid_helper.update_lim(self.axes)
            #self.set_transform(self._grid_helper.get_gridlines_transform())
            gl = self._grid_helper.get_gridlines()
            if gl:
                self.set_segments([np.transpose(l) for l in gl])
            else:
                self.set_segments([])
        super(GridlinesCollection, self).draw(renderer)



class AxisGridLineBase(martist.Artist):
    def __init__(self, *kl, **kw):
        super(AxisGridLineBase, self).__init__(*kl, **kw)


class AxisArtist(AxisGridLineBase):
    """
    an artist which draws axis (a line along which the n-th axes coord
    is constant) line, ticks, ticklabels, and axis label.

    It requires an AxisArtistHelper instance.
    """

    LABELPAD = 5
    ZORDER=2.5

    def __init__(self, axes,
                 helper,
                 #offset_transform=None,
                 offset=None,
                 major_tick_size=None,
                 major_tick_pad=None,
                 minor_tick_size=None,
                 minor_tick_pad=None,
                 **kw):
        """
        axes is also used to follow the axis attribute (tick color, etc).
        """
        AxisGridLineBase.__init__(self, **kw)

        self.axes = axes

        self._axis_artist_helper = helper

        if offset is None:
            offset = (0, 0)
        self.dpi_transform = Affine2D()
        self.offset_transform = ScaledTranslation(offset[0], offset[1],
                                                  self.dpi_transform)

        #self.set_transform(axes.transAxes + \
        #                   self.offset_transform)

        self._label_visible = True
        self._majortick_visible = True
        self._majorticklabel_visible = True
        self._minortick_visible = True
        self._minorticklabel_visible = True


        if self._axis_artist_helper.label_direction in ["left", "right"]:
            axis_name = "ytick"
            self.axis = axes.yaxis
        else:
            axis_name = "xtick"
            self.axis = axes.xaxis


        if major_tick_size is None:
            self.major_tick_size = rcParams['%s.major.size'%axis_name]
        if major_tick_pad is None:
            self.major_tick_pad = rcParams['%s.major.pad'%axis_name]
        if minor_tick_size is None:
            self.minor_tick_size = rcParams['%s.minor.size'%axis_name]
        if minor_tick_pad is None:
            self.minor_tick_pad = rcParams['%s.minor.pad'%axis_name]

        self._init_line()
        self._init_ticks()
        self._init_offsetText(self._axis_artist_helper.label_direction)
        self._init_label()

        self.set_zorder(self.ZORDER)

    def get_transform(self):
        return self.axes.transAxes + self.offset_transform

    def get_helper(self):
        return self._axis_artist_helper


    def _init_line(self):
        tran = self._axis_artist_helper.get_line_transform(self.axes) \
               + self.offset_transform
        self.line = BezierPath(self._axis_artist_helper.get_line(self.axes),
                               color=rcParams['axes.edgecolor'],
                               linewidth=rcParams['axes.linewidth'],
                               transform=tran)

    def _draw_line(self, renderer):
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        self.line.draw(renderer)


    def _init_ticks(self):

        transform=self._axis_artist_helper.get_tick_transform(self.axes) \
                   + self.offset_transform

        self.major_ticks = Ticks(self.major_tick_size,
                                 axis=self.axis,
                                 transform=transform)
        self.minor_ticks = Ticks(self.minor_tick_size,
                                 axis=self.axis,
                                 transform=transform)


        size = rcParams['xtick.labelsize']

        fontprops = font_manager.FontProperties(size=size)
        #tvhl = self._axis_artist_helper.get_ticklabel_transform(
        tvhl = self._axis_artist_helper.get_ticklabel_offset_transform( \
            self.axes,
            self.major_tick_pad,
            fontprops=fontprops,
            renderer=None,
            )
        #trans=transform)
        trans, vert, horiz, label_a = tvhl
        trans = transform + trans

        self.major_ticklabels = TickLabels(size, axis=self.axis)
        self.minor_ticklabels = TickLabels(size, axis=self.axis)


        self.major_ticklabels.set(figure = self.axes.figure,
                                  rotation = label_a,
                                  transform=trans,
                                  va=vert,
                                  ha=horiz,
                                  fontproperties=fontprops)

        self.minor_ticklabels.set(figure = self.axes.figure,
                                  rotation = label_a,
                                  transform=trans,
                                  va=vert,
                                  ha=horiz,
                                  fontproperties=fontprops)


    _offsetText_pos = dict(left=(0, 1, "bottom", "right"),
                           right=(1, 1, "bottom", "left"),
                           bottom=(1, 0, "top", "right"),
                           top=(1, 1, "bottom", "right"))

    def _init_offsetText(self, direction):

        x,y,va,ha = self._offsetText_pos[direction]

        #d = self._axis_artist_helper.label_direction
        #fp = font_manager.FontProperties(size=rcParams['xtick.labelsize'])
        #fp = font_manager.FontProperties(size=self.major_ticklabels.get_size())
        self.offsetText = mtext.Annotation("",
                                           xy=(x,y), xycoords="axes fraction",
                                           xytext=(0,0), textcoords="offset points",
                                           #fontproperties = fp,
                                           color = rcParams['xtick.color'],
                                           verticalalignment=va,
                                           horizontalalignment=ha,
                                           )
        self.offsetText.set_transform(IdentityTransform())
        self.axes._set_artist_props(self.offsetText)


    def _update_offsetText(self):
        self.offsetText.set_text( self.axis.major.formatter.get_offset() )
        self.offsetText.set_size(self.major_ticklabels.get_size())
        offset = self.major_tick_pad + self.major_ticklabels.get_size() + 2.
        self.offsetText.xytext= (0, offset)


    def _draw_offsetText(self, renderer):
        self._update_offsetText()
        self.offsetText.draw(renderer)


    def _draw_ticks(self, renderer):
        #majortick_iter, minortick_iter):
        #major_locs, major_angles,
        #minor_locs, minor_angles):

        majortick_iter,  minortick_iter = \
                        self._axis_artist_helper.get_tick_iterators(self.axes)

        tick_loc_angles = []
        tick_loc_labels = []
        for tick_loc, tick_angle, tick_label in majortick_iter:
            tick_loc_angles.append((tick_loc, tick_angle))
            tick_loc_labels.append((tick_loc, tick_label))


        transform=self._axis_artist_helper.get_tick_transform(self.axes) \
                   + self.offset_transform
        fontprops = font_manager.FontProperties(size=12)
        tvhl = self._axis_artist_helper.get_ticklabel_offset_transform( \
            self.axes,
            self.major_tick_pad,
            fontprops=fontprops,
            renderer=renderer,
            )
        #trans=transform)
        trans, va, ha, a = tvhl
        trans = transform + trans

        self.major_ticklabels.set(transform=trans,
                                  va=va, ha=ha, rotation=a)


        self.major_ticks.update_locs_angles(tick_loc_angles, renderer)
        self.major_ticklabels.update_locs_labels(tick_loc_labels, renderer)

        self.major_ticks.draw(renderer)
        self.major_ticklabels.draw(renderer)

        tick_loc_angles = []
        tick_loc_labels = []
        for tick_loc, tick_angle, tick_label in minortick_iter:
            tick_loc_angles.append((tick_loc, tick_angle))
            tick_loc_labels.append((tick_loc, tick_label))

        self.minor_ticks.update_locs_angles(tick_loc_angles, renderer)
        self.minor_ticklabels.update_locs_labels(tick_loc_labels, renderer)

        self.minor_ticks.draw(renderer)
        self.minor_ticklabels.draw(renderer)

        if (self.major_ticklabels.get_visible() or self.minor_ticklabels.get_visible()):
            self._draw_offsetText(renderer)

        return self.major_ticklabels.get_window_extents(renderer)

    def _init_label(self):
        # x in axes coords, y in display coords (to be updated at draw
        # time by _update_label_positions)
        fontprops = font_manager.FontProperties(size=rcParams['axes.labelsize'])
        textprops = dict(fontproperties = fontprops,
                         color = rcParams['axes.labelcolor'],
                         )

        self.label = AxisLabel(0, 0, "__from_axes__",
                               color = "auto", #rcParams['axes.labelcolor'],
                               fontproperties=fontprops,
                               axis=self.axis,
                               )

        self.label.set_figure(self.axes.figure)

        #self._set_artist_props(label)

    def _draw_label(self, renderer, bboxes):

        if not self.label.get_visible():
            return

        fontprops = font_manager.FontProperties(size=rcParams['axes.labelsize'])
        pad_points = self.LABELPAD + self.major_tick_pad
        xy, tr = self._axis_artist_helper.get_label_pos(self.axes)
        if xy is None: return

        x, y = xy
        tr2, va, ha, a = self._axis_artist_helper.get_label_offset_transform(\
            self.axes,
            pad_points, fontprops,
            renderer,
            bboxes=bboxes,
            )
                                                          #trans=tr+self.offset_transform)
        tr2 = (tr+self.offset_transform) + tr2

        self.label.set(x=x, y=y,
                       transform=tr2,
                       va=va, ha=ha, rotation=a)

#         if self.label.get_text() == "__from_axes__":
#             label_text = self._helper.axis.get_label().get_text()
#             self.label.set_text(label_text)
#             self.label.draw(renderer)
#             self.label.set_text("__from_axes__")
#         else:

        self.label.draw(renderer)


    def set_label(self, s):
        self.label.set_text(s)


    def draw(self, renderer):
        'Draw the axis lines, tick lines and labels'

        if not self.get_visible(): return

        renderer.open_group(__name__)

        self._axis_artist_helper.update_lim(self.axes)

        dpi_cor = renderer.points_to_pixels(1.)
        self.dpi_transform.clear().scale(dpi_cor, dpi_cor)


        self._draw_line(renderer)
        bboxes = self._draw_ticks(renderer)

        #self._draw_offsetText(renderer)
        self._draw_label(renderer, bboxes)

        renderer.close_group(__name__)

    def get_ticklabel_extents(self, renderer):
        pass

    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):
        if all:
            _ticks, _ticklabels, _label = True, True, True
        elif all is not None:
            _ticks, _ticklabels, _label = False, False, False
        else:
            _ticks, _ticklabels, _label = None, None, None

        if ticks is not None:
            _ticks = ticks
        if ticklabels is not None:
            _ticklabels = ticklabels
        if label is not None:
            _label = label

        if _ticks is not None:
            self.major_ticks.set_visible(_ticks)
            self.minor_ticks.set_visible(_ticks)
        if _ticklabels is not None:
            self.major_ticklabels.set_visible(_ticklabels)
            self.minor_ticklabels.set_visible(_ticklabels)
        if _label is not None:
            self.label.set_visible(_label)


class Axes(maxes.Axes):

    class AxisDict(dict):
        def __init__(self, axes):
            self.axes = axes
            super(Axes.AxisDict, self).__init__()

        def __call__(self, *v, **kwargs):
            return maxes.Axes.axis(self.axes, *v, **kwargs)


    def __init__(self, *kl, **kw):


        helper = kw.pop("grid_helper", None)

        if helper:
            self._grid_helper = helper
        else:
            self._grid_helper = GridHelperRectlinear(self)

        #if self._grid_helper.axes is None:
        #    self._grid_helper.set_axes(self)
        self._axisline_on = True

        super(Axes, self).__init__(*kl, **kw)

        self.toggle_axisline(True)


    def toggle_axisline(self, b=None):
        if b is None:
            b = not self._axisline_on
        if b:
            self._axisline_on = True
            for s in self.spines.values():
                s.set_visible(False)
            self.xaxis.set_visible(False)
            self.yaxis.set_visible(False)
        else:
            self._axisline_on = False
            for s in self.spines.values():
                s.set_visible(True)
            self.xaxis.set_visible(True)
            self.yaxis.set_visible(True)


    def _init_axis(self):
        super(Axes, self)._init_axis()


    def _init_axislines(self):
        self._axislines = self.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        for loc in ["bottom", "top", "left", "right"]:
            self._axislines[loc] = new_fixed_axis(loc=loc, axes=self)

        for axisline in [self._axislines["top"], self._axislines["right"]]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    def _get_axislines(self):
        return self._axislines

    axis = property(_get_axislines)

    def new_gridlines(self, grid_helper=None):
        gridlines = GridlinesCollection(None, transform=self.transData,
                                        colors=rcParams['grid.color'],
                                        linestyles=rcParams['grid.linestyle'],
                                        linewidths=rcParams['grid.linewidth'])
        self._set_artist_props(gridlines)
        if grid_helper is None:
            grid_helper = self.get_grid_helper()
        gridlines.set_grid_helper(grid_helper)
        gridlines.set_clip_on(True)

        return gridlines

    def cla(self):
        # gridlines need to b created before cla() since cla calls grid()
        self.gridlines = self.new_gridlines()

        super(Axes, self).cla()
        self._init_axislines()

    def get_grid_helper(self):
        return self._grid_helper


    def grid(self, b=None, **kwargs):
        if not self._axisline_on:
            super(Axes, self).grid(b, **kwargs)
            return

        if b is None:
            b = not self.gridlines.get_visible()

        self.gridlines.set_visible(b)

        if len(kwargs):
            martist.setp(self.gridlines, **kwargs)

    #def get_gridlines(self):
    #    return self._grid_helper.get_gridlines()

    def get_children(self):
        if self._axisline_on:
            children = self._axislines.values()+[self.gridlines]
        else:
            children = []
        children.extend(super(Axes, self).get_children())
        return children

    def invalidate_grid_helper(self):
        #self._grid_helper.update_lim(self, force_update=True)
        self._grid_helper.invalidate()


    def draw(self, renderer, inframe=False):

        if not self._axisline_on:
            super(Axes, self).draw(renderer, inframe)
            return

        orig_artists = self.artists
        self.artists = self.artists + list(self._axislines.values()) + [self.gridlines]

        super(Axes, self).draw(renderer, inframe)

        self.artists = orig_artists


    def get_tightbbox(self, renderer):

        bb0 = super(Axes, self).get_tightbbox(renderer)

        if not self._axisline_on:
            return bb0


        #artists = []
        bb = [bb0]

        for axisline in self._axislines.values():
            if not axisline.get_visible():
                continue

            if axisline.label.get_visible():
                bb.append(axisline.label.get_window_extent(renderer))


            if axisline.major_ticklabels.get_visible():
                bb.extend(axisline.major_ticklabels.get_window_extents(renderer))
            if axisline.minor_ticklabels.get_visible():
                bb.extend(axisline.minor_ticklabels.get_window_extents(renderer))
            if axisline.major_ticklabels.get_visible() or \
               axisline.minor_ticklabels.get_visible():
                bb.append(axisline.offsetText.get_window_extent(renderer))

        #bb.extend([c.get_window_extent(renderer) for c in artists \
        #           if c.get_visible()])

        _bbox = Bbox.union([b for b in bb if b.width!=0 or b.height!=0])

        return _bbox



Subplot = maxes.subplot_class_factory(Axes)


class AxesZero(Axes):
    def __init__(self, *kl, **kw):

        super(AxesZero, self).__init__(*kl, **kw)


    def _init_axislines(self):
        super(AxesZero, self)._init_axislines()

        new_floating_axis = self._grid_helper.new_floating_axis
        xaxis_zero = new_floating_axis(nth_coord=0,
                                       passthrough_point=(0.,0.),
                                       transform=self.transData,
                                       tick_direction="in",
                                       label_direction="bottom",
                                       axes=self)

        xaxis_zero.line.set_clip_path(self.patch)
        xaxis_zero.set_visible(False)
        self._axislines["xzero"] = xaxis_zero

        yaxis_zero = new_floating_axis(nth_coord=1,
                                       passthrough_point=(0.,0.),
                                       transform=self.transData,
                                       tick_direction="in",
                                       label_direction="left",
                                       axes=self)


        yaxis_zero.line.set_clip_path(self.patch)
        yaxis_zero.set_visible(False)
        self._axislines["yzero"] = yaxis_zero

SubplotZero = maxes.subplot_class_factory(AxesZero)


if __name__ == "__main__":
    fig = plt.figure(1, (4,3))

    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    ax.axis["xzero"].set_visible(True)
    ax.axis["xzero"].label.set_text("Axis Zero")

    for n in ["bottom", "top", "right"]:
        ax.axis[n].set_visible(False)

    xx = np.arange(0, 2*np.pi, 0.01)
    ax.plot(xx, np.sin(xx))

    plt.show()


