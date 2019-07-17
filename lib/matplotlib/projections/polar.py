from collections import OrderedDict
import types

import numpy as np

from matplotlib import cbook, rcParams
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.spines as mspines


class PolarTransform(mtransforms.Transform):
    """
    The base polar transform.  This handles projection *theta* and
    *r* into Cartesian coordinate space *x* and *y*, but does not
    perform the ultimate affine transformation into the correct
    position.
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, axis=None, use_rmin=True,
                 _apply_theta_transforms=True):
        mtransforms.Transform.__init__(self)
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = _apply_theta_transforms

    def __str__(self):
        return ("{}(\n"
                    "{},\n"
                "    use_rmin={},\n"
                "    _apply_theta_transforms={})"
                .format(type(self).__name__,
                        mtransforms._indent_str(self._axis),
                        self._use_rmin,
                        self._apply_theta_transforms))

    def transform_non_affine(self, tr):
        # docstring inherited
        t, r = np.transpose(tr)
        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            t *= self._axis.get_theta_direction()
            t += self._axis.get_theta_offset()
        if self._use_rmin and self._axis is not None:
            r = (r - self._axis.get_rorigin()) * self._axis.get_rsign()
        r = np.where(r >= 0, r, np.nan)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])

    def transform_path_non_affine(self, path):
        # docstring inherited
        vertices = path.vertices
        if len(vertices) == 2 and vertices[0, 0] == vertices[1, 0]:
            return mpath.Path(self.transform(vertices), path.codes)
        ipath = path.interpolated(path._interpolation_steps)
        return mpath.Path(self.transform(ipath.vertices), ipath.codes)

    def inverted(self):
        # docstring inherited
        return PolarAxes.InvertedPolarTransform(self._axis, self._use_rmin,
                                                self._apply_theta_transforms)


class PolarAffine(mtransforms.Affine2DBase):
    """
    The affine part of the polar projection.  Scales the output so
    that maximum radius rests on the edge of the axes circle.
    """
    def __init__(self, scale_transform, limits):
        """
        *limits* is the view limit of the data.  The only part of
        its bounds that is used is the y limits (for the radius limits).
        The theta range is handled by the non-affine transform.
        """
        mtransforms.Affine2DBase.__init__(self)
        self._scale_transform = scale_transform
        self._limits = limits
        self.set_children(scale_transform, limits)
        self._mtx = None

    def __str__(self):
        return ("{}(\n"
                    "{},\n"
                    "{})"
                .format(type(self).__name__,
                        mtransforms._indent_str(self._scale_transform),
                        mtransforms._indent_str(self._limits)))

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            limits_scaled = self._limits.transformed(self._scale_transform)
            yscale = limits_scaled.ymax - limits_scaled.ymin
            affine = mtransforms.Affine2D() \
                .scale(0.5 / yscale) \
                .translate(0.5, 0.5)
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        return self._mtx


class InvertedPolarTransform(mtransforms.Transform):
    """
    The inverse of the polar transform, mapping Cartesian
    coordinate space *x* and *y* back to *theta* and *r*.
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, axis=None, use_rmin=True,
                 _apply_theta_transforms=True):
        mtransforms.Transform.__init__(self)
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = _apply_theta_transforms

    def __str__(self):
        return ("{}(\n"
                    "{},\n"
                "    use_rmin={},\n"
                "    _apply_theta_transforms={})"
                .format(type(self).__name__,
                        mtransforms._indent_str(self._axis),
                        self._use_rmin,
                        self._apply_theta_transforms))

    def transform_non_affine(self, xy):
        # docstring inherited
        x = xy[:, 0:1]
        y = xy[:, 1:]
        r = np.hypot(x, y)
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)

        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            theta -= self._axis.get_theta_offset()
            theta *= self._axis.get_theta_direction()
            theta %= 2 * np.pi

        if self._use_rmin and self._axis is not None:
            r += self._axis.get_rorigin()
            r *= self._axis.get_rsign()

        return np.concatenate((theta, r), 1)

    def inverted(self):
        # docstring inherited
        return PolarAxes.PolarTransform(self._axis, self._use_rmin,
                                        self._apply_theta_transforms)


class ThetaFormatter(mticker.Formatter):
    """
    Used to format the *theta* tick labels.  Converts the native
    unit of radians into degrees and adds a degree symbol.
    """
    def __call__(self, x, pos=None):
        vmin, vmax = self.axis.get_view_interval()
        d = np.rad2deg(abs(vmax - vmin))
        digits = max(-int(np.log10(d) - 1.5), 0)

        if rcParams['text.usetex'] and not rcParams['text.latex.unicode']:
            format_str = r"${value:0.{digits:d}f}^\circ$"
            return format_str.format(value=np.rad2deg(x), digits=digits)
        else:
            # we use unicode, rather than mathtext with \circ, so
            # that it will work correctly with any arbitrary font
            # (assuming it has a degree sign), whereas $5\circ$
            # will only work correctly with one of the supported
            # math fonts (Computer Modern and STIX)
            format_str = "{value:0.{digits:d}f}\N{DEGREE SIGN}"
            return format_str.format(value=np.rad2deg(x), digits=digits)


class _AxisWrapper(object):
    def __init__(self, axis):
        self._axis = axis

    def get_view_interval(self):
        return np.rad2deg(self._axis.get_view_interval())

    def set_view_interval(self, vmin, vmax):
        self._axis.set_view_interval(*np.deg2rad((vmin, vmax)))

    def get_minpos(self):
        return np.rad2deg(self._axis.get_minpos())

    def get_data_interval(self):
        return np.rad2deg(self._axis.get_data_interval())

    def set_data_interval(self, vmin, vmax):
        self._axis.set_data_interval(*np.deg2rad((vmin, vmax)))

    def get_tick_space(self):
        return self._axis.get_tick_space()


class ThetaLocator(mticker.Locator):
    """
    Used to locate theta ticks.

    This will work the same as the base locator except in the case that the
    view spans the entire circle. In such cases, the previously used default
    locations of every 45 degrees are returned.
    """
    def __init__(self, base):
        self.base = base
        self.axis = self.base.axis = _AxisWrapper(self.base.axis)

    def set_axis(self, axis):
        self.axis = _AxisWrapper(axis)
        self.base.set_axis(self.axis)

    def __call__(self):
        lim = self.axis.get_view_interval()
        if _is_full_circle_deg(lim[0], lim[1]):
            return np.arange(8) * 2 * np.pi / 8
        else:
            return np.deg2rad(self.base())

    def autoscale(self):
        return self.base.autoscale()

    def pan(self, numsteps):
        return self.base.pan(numsteps)

    def refresh(self):
        return self.base.refresh()

    def view_limits(self, vmin, vmax):
        vmin, vmax = np.rad2deg((vmin, vmax))
        return np.deg2rad(self.base.view_limits(vmin, vmax))

    def zoom(self, direction):
        return self.base.zoom(direction)


class ThetaTick(maxis.XTick):
    """
    A theta-axis tick.

    This subclass of `XTick` provides angular ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    tick location. This results in ticks that are correctly perpendicular to
    the arc spine.

    When 'auto' rotation is enabled, labels are also rotated to be parallel to
    the spine. The label padding is also applied here since it's not possible
    to use a generic axes transform to produce tick-specific padding.
    """
    def __init__(self, axes, *args, **kwargs):
        self._text1_translate = mtransforms.ScaledTranslation(
            0, 0,
            axes.figure.dpi_scale_trans)
        self._text2_translate = mtransforms.ScaledTranslation(
            0, 0,
            axes.figure.dpi_scale_trans)
        super().__init__(axes, *args, **kwargs)

    def _get_text1(self):
        t = super()._get_text1()
        t.set_rotation_mode('anchor')
        t.set_transform(t.get_transform() + self._text1_translate)
        return t

    def _get_text2(self):
        t = super()._get_text2()
        t.set_rotation_mode('anchor')
        t.set_transform(t.get_transform() + self._text2_translate)
        return t

    def _apply_params(self, **kw):
        super()._apply_params(**kw)

        # Ensure transform is correct; sometimes this gets reset.
        trans = self.label1.get_transform()
        if not trans.contains_branch(self._text1_translate):
            self.label1.set_transform(trans + self._text1_translate)
        trans = self.label2.get_transform()
        if not trans.contains_branch(self._text2_translate):
            self.label2.set_transform(trans + self._text2_translate)

    def _update_padding(self, pad, angle):
        padx = pad * np.cos(angle) / 72
        pady = pad * np.sin(angle) / 72
        self._text1_translate._t = (padx, pady)
        self._text1_translate.invalidate()
        self._text2_translate._t = (-padx, -pady)
        self._text2_translate.invalidate()

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        angle = loc * axes.get_theta_direction() + axes.get_theta_offset()
        text_angle = np.rad2deg(angle) % 360 - 90
        angle -= np.pi / 2

        marker = self.tick1line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        marker = self.tick2line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans

        mode, user_angle = self._labelrotation
        if mode == 'default':
            text_angle = user_angle
        else:
            if text_angle > 90:
                text_angle -= 180
            elif text_angle < -90:
                text_angle += 180
            text_angle += user_angle
        self.label1.set_rotation(text_angle)
        self.label2.set_rotation(text_angle)

        # This extra padding helps preserve the look from previous releases but
        # is also needed because labels are anchored to their center.
        pad = self._pad + 7
        self._update_padding(pad,
                             self._loc * axes.get_theta_direction() +
                             axes.get_theta_offset())


class ThetaAxis(maxis.XAxis):
    """
    A theta Axis.

    This overrides certain properties of an `XAxis` to provide special-casing
    for an angular axis.
    """
    __name__ = 'thetaaxis'
    axis_name = 'theta'

    def _get_tick(self, major):
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return ThetaTick(self.axes, 0, '', major=major, **tick_kw)

    def _wrap_locator_formatter(self):
        self.set_major_locator(ThetaLocator(self.get_major_locator()))
        self.set_major_formatter(ThetaFormatter())
        self.isDefault_majloc = True
        self.isDefault_majfmt = True

    def cla(self):
        super().cla()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        super()._set_scale(value, **kwargs)
        self._wrap_locator_formatter()

    def _copy_tick_props(self, src, dest):
        'Copy the props from src tick to dest tick'
        if src is None or dest is None:
            return
        super()._copy_tick_props(src, dest)

        # Ensure that tick transforms are independent so that padding works.
        trans = dest._get_text1_transform()[0]
        dest.label1.set_transform(trans + dest._text1_translate)
        trans = dest._get_text2_transform()[0]
        dest.label2.set_transform(trans + dest._text2_translate)


class RadialLocator(mticker.Locator):
    """
    Used to locate radius ticks.

    Ensures that all ticks are strictly positive.  For all other
    tasks, it delegates to the base
    :class:`~matplotlib.ticker.Locator` (which may be different
    depending on the scale of the *r*-axis.
    """
    def __init__(self, base, axes=None):
        self.base = base
        self._axes = axes

    def __call__(self):
        show_all = True
        # Ensure previous behaviour with full circle non-annular views.
        if self._axes:
            if _is_full_circle_rad(*self._axes.viewLim.intervalx):
                rorigin = self._axes.get_rorigin() * self._axes.get_rsign()
                if self._axes.get_rmin() <= rorigin:
                    show_all = False
        if show_all:
            return self.base()
        else:
            return [tick for tick in self.base() if tick > rorigin]

    def autoscale(self):
        return self.base.autoscale()

    def pan(self, numsteps):
        return self.base.pan(numsteps)

    def zoom(self, direction):
        return self.base.zoom(direction)

    def refresh(self):
        return self.base.refresh()

    def view_limits(self, vmin, vmax):
        vmin, vmax = self.base.view_limits(vmin, vmax)
        if vmax > vmin:
            # this allows inverted r/y-lims
            vmin = min(0, vmin)
        return mtransforms.nonsingular(vmin, vmax)


class _ThetaShift(mtransforms.ScaledTranslation):
    """
    Apply a padding shift based on axes theta limits.

    This is used to create padding for radial ticks.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The owning axes; used to determine limits.
    pad : float
        The padding to apply, in points.
    mode : {'min', 'max', 'rlabel'}
        Whether to shift away from the start (``'min'``) or the end (``'max'``)
        of the axes, or using the rlabel position (``'rlabel'``).
    """
    def __init__(self, axes, pad, mode):
        mtransforms.ScaledTranslation.__init__(self, pad, pad,
                                               axes.figure.dpi_scale_trans)
        self.set_children(axes._realViewLim)
        self.axes = axes
        self.mode = mode
        self.pad = pad

    def __str__(self):
        return ("{}(\n"
                    "{},\n"
                    "{},\n"
                    "{})"
                .format(type(self).__name__,
                        mtransforms._indent_str(self.axes),
                        mtransforms._indent_str(self.pad),
                        mtransforms._indent_str(repr(self.mode))))

    def get_matrix(self):
        if self._invalid:
            if self.mode == 'rlabel':
                angle = (
                    np.deg2rad(self.axes.get_rlabel_position()) *
                    self.axes.get_theta_direction() +
                    self.axes.get_theta_offset()
                )
            else:
                if self.mode == 'min':
                    angle = self.axes._realViewLim.xmin
                elif self.mode == 'max':
                    angle = self.axes._realViewLim.xmax

            if self.mode in ('rlabel', 'min'):
                padx = np.cos(angle - np.pi / 2)
                pady = np.sin(angle - np.pi / 2)
            else:
                padx = np.cos(angle + np.pi / 2)
                pady = np.sin(angle + np.pi / 2)

            self._t = (self.pad * padx / 72, self.pad * pady / 72)
        return mtransforms.ScaledTranslation.get_matrix(self)


class RadialTick(maxis.YTick):
    """
    A radial-axis tick.

    This subclass of `YTick` provides radial ticks with some small modification
    to their re-positioning such that ticks are rotated based on axes limits.
    This results in ticks that are correctly perpendicular to the spine. Labels
    are also rotated to be perpendicular to the spine, when 'auto' rotation is
    enabled.
    """
    def _get_text1(self):
        t = super()._get_text1()
        t.set_rotation_mode('anchor')
        return t

    def _get_text2(self):
        t = super()._get_text2()
        t.set_rotation_mode('anchor')
        return t

    def _determine_anchor(self, mode, angle, start):
        # Note: angle is the (spine angle - 90) because it's used for the tick
        # & text setup, so all numbers below are -90 from (normed) spine angle.
        if mode == 'auto':
            if start:
                if -90 <= angle <= 90:
                    return 'left', 'center'
                else:
                    return 'right', 'center'
            else:
                if -90 <= angle <= 90:
                    return 'right', 'center'
                else:
                    return 'left', 'center'
        else:
            if start:
                if angle < -68.5:
                    return 'center', 'top'
                elif angle < -23.5:
                    return 'left', 'top'
                elif angle < 22.5:
                    return 'left', 'center'
                elif angle < 67.5:
                    return 'left', 'bottom'
                elif angle < 112.5:
                    return 'center', 'bottom'
                elif angle < 157.5:
                    return 'right', 'bottom'
                elif angle < 202.5:
                    return 'right', 'center'
                elif angle < 247.5:
                    return 'right', 'top'
                else:
                    return 'center', 'top'
            else:
                if angle < -68.5:
                    return 'center', 'bottom'
                elif angle < -23.5:
                    return 'right', 'bottom'
                elif angle < 22.5:
                    return 'right', 'center'
                elif angle < 67.5:
                    return 'right', 'top'
                elif angle < 112.5:
                    return 'center', 'top'
                elif angle < 157.5:
                    return 'left', 'top'
                elif angle < 202.5:
                    return 'left', 'center'
                elif angle < 247.5:
                    return 'left', 'bottom'
                else:
                    return 'center', 'bottom'

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        thetamin = axes.get_thetamin()
        thetamax = axes.get_thetamax()
        direction = axes.get_theta_direction()
        offset_rad = axes.get_theta_offset()
        offset = np.rad2deg(offset_rad)
        full = _is_full_circle_deg(thetamin, thetamax)

        if full:
            angle = (axes.get_rlabel_position() * direction +
                     offset) % 360 - 90
            tick_angle = 0
        else:
            angle = (thetamin * direction + offset) % 360 - 90
            if direction > 0:
                tick_angle = np.deg2rad(angle)
            else:
                tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        if full:
            ha = self.label1.get_horizontalalignment()
            va = self.label1.get_verticalalignment()
        else:
            ha, va = self._determine_anchor(mode, angle, direction > 0)
        self.label1.set_horizontalalignment(ha)
        self.label1.set_verticalalignment(va)
        self.label1.set_rotation(text_angle)

        marker = self.tick1line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        if full:
            self.label2.set_visible(False)
            self.tick2line.set_visible(False)
        angle = (thetamax * direction + offset) % 360 - 90
        if direction > 0:
            tick_angle = np.deg2rad(angle)
        else:
            tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        ha, va = self._determine_anchor(mode, angle, direction < 0)
        self.label2.set_ha(ha)
        self.label2.set_va(va)
        self.label2.set_rotation(text_angle)

        marker = self.tick2line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans


class RadialAxis(maxis.YAxis):
    """
    A radial Axis.

    This overrides certain properties of a `YAxis` to provide special-casing
    for a radial axis.
    """
    __name__ = 'radialaxis'
    axis_name = 'radius'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sticky_edges.y.append(0)

    def _get_tick(self, major):
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return RadialTick(self.axes, 0, '', major=major, **tick_kw)

    def _wrap_locator_formatter(self):
        self.set_major_locator(RadialLocator(self.get_major_locator(),
                                             self.axes))
        self.isDefault_majloc = True

    def cla(self):
        super().cla()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        super()._set_scale(value, **kwargs)
        self._wrap_locator_formatter()


def _is_full_circle_deg(thetamin, thetamax):
    """
    Determine if a wedge (in degrees) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 360.0) < 1e-12


def _is_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 2 * np.pi) < 1.74e-14


class _WedgeBbox(mtransforms.Bbox):
    """
    Transform (theta,r) wedge Bbox into axes bounding box.

    Parameters
    ----------
    center : (float, float)
        Center of the wedge
    viewLim : `~matplotlib.transforms.Bbox`
        Bbox determining the boundaries of the wedge
    originLim : `~matplotlib.transforms.Bbox`
        Bbox determining the origin for the wedge, if different from *viewLim*
    """
    def __init__(self, center, viewLim, originLim, **kwargs):
        mtransforms.Bbox.__init__(self,
                                  np.array([[0.0, 0.0], [1.0, 1.0]], np.float),
                                  **kwargs)
        self._center = center
        self._viewLim = viewLim
        self._originLim = originLim
        self.set_children(viewLim, originLim)

    def __str__(self):
        return ("{}(\n"
                    "{},\n"
                    "{},\n"
                    "{})"
                .format(type(self).__name__,
                        mtransforms._indent_str(self._center),
                        mtransforms._indent_str(self._viewLim),
                        mtransforms._indent_str(self._originLim)))

    def get_points(self):
        # docstring inherited
        if self._invalid:
            points = self._viewLim.get_points().copy()
            # Scale angular limits to work with Wedge.
            points[:, 0] *= 180 / np.pi
            if points[0, 0] > points[1, 0]:
                points[:, 0] = points[::-1, 0]

            # Scale radial limits based on origin radius.
            points[:, 1] -= self._originLim.y0

            # Scale radial limits to match axes limits.
            rscale = 0.5 / points[1, 1]
            points[:, 1] *= rscale
            width = min(points[1, 1] - points[0, 1], 0.5)

            # Generate bounding box for wedge.
            wedge = mpatches.Wedge(self._center, points[1, 1],
                                   points[0, 0], points[1, 0],
                                   width=width)
            self.update_from_path(wedge.get_path())

            # Ensure equal aspect ratio.
            w, h = self._points[1] - self._points[0]
            deltah = max(w - h, 0) / 2
            deltaw = max(h - w, 0) / 2
            self._points += np.array([[-deltaw, -deltah], [deltaw, deltah]])

            self._invalid = 0

        return self._points


class PolarAxes(Axes):
    """
    A polar graph projection, where the input dimensions are *theta*, *r*.

    Theta starts pointing east and goes anti-clockwise.
    """
    name = 'polar'

    def __init__(self, *args,
                 theta_offset=0, theta_direction=1, rlabel_position=22.5,
                 **kwargs):
        # docstring inherited
        self._default_theta_offset = theta_offset
        self._default_theta_direction = theta_direction
        self._default_rlabel_position = np.deg2rad(rlabel_position)
        super().__init__(*args, **kwargs)
        self.use_sticky_edges = True
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()

    def cla(self):
        Axes.cla(self)

        self.title.set_y(1.05)

        start = self.spines.get('start', None)
        if start:
            start.set_visible(False)
        end = self.spines.get('end', None)
        if end:
            end.set_visible(False)
        self.set_xlim(0.0, 2 * np.pi)

        self.grid(rcParams['polaraxes.grid'])
        inner = self.spines.get('inner', None)
        if inner:
            inner.set_visible(False)

        self.set_rorigin(None)
        self.set_theta_offset(self._default_theta_offset)
        self.set_theta_direction(self._default_theta_direction)

    def _init_axis(self):
        "move this out of __init__ because non-separable axes don't use it"
        self.xaxis = ThetaAxis(self)
        self.yaxis = RadialAxis(self)
        # Calling polar_axes.xaxis.cla() or polar_axes.xaxis.cla()
        # results in weird artifacts. Therefore we disable this for
        # now.
        # self.spines['polar'].register_axis(self.yaxis)
        self._update_transScale()

    def _set_lim_and_transforms(self):
        # A view limit where the minimum radius can be locked if the user
        # specifies an alternate origin.
        self._originViewLim = mtransforms.LockableBbox(self.viewLim)

        # Handle angular offset and direction.
        self._direction = mtransforms.Affine2D() \
            .scale(self._default_theta_direction, 1.0)
        self._theta_offset = mtransforms.Affine2D() \
            .translate(self._default_theta_offset, 0.0)
        self.transShift = mtransforms.composite_transform_factory(
            self._direction,
            self._theta_offset)
        # A view limit shifted to the correct location after accounting for
        # orientation and offset.
        self._realViewLim = mtransforms.TransformedBbox(self.viewLim,
                                                        self.transShift)

        # Transforms the x and y axis separately by a scale factor
        # It is assumed that this part will have non-linear components
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # Scale view limit into a bbox around the selected wedge. This may be
        # smaller than the usual unit axes rectangle if not plotting the full
        # circle.
        self.axesLim = _WedgeBbox((0.5, 0.5),
                                  self._realViewLim, self._originViewLim)

        # Scale the wedge to fill the axes.
        self.transWedge = mtransforms.BboxTransformFrom(self.axesLim)

        # Scale the axes to fill the figure.
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # A (possibly non-linear) projection on the (already scaled)
        # data.  This one is aware of rmin
        self.transProjection = self.PolarTransform(
            self,
            _apply_theta_transforms=False)
        # Add dependency on rorigin.
        self.transProjection.set_children(self._originViewLim)

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transProjectionAffine = self.PolarAffine(self.transScale,
                                                      self._originViewLim)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        self.transData = (
            self.transScale + self.transShift + self.transProjection +
            (self.transProjectionAffine + self.transWedge + self.transAxes))

        # This is the transform for theta-axis ticks.  It is
        # equivalent to transData, except it always puts r == 0.0 and r == 1.0
        # at the edge of the axis circles.
        self._xaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.IdentityTransform(),
                mtransforms.BboxTransformTo(self.viewLim)) +
            self.transData)
        # The theta labels are flipped along the radius, so that text 1 is on
        # the outside by default. This should work the same as before.
        flipr_transform = mtransforms.Affine2D() \
            .translate(0.0, -0.5) \
            .scale(1.0, -1.0) \
            .translate(0.0, 0.5)
        self._xaxis_text_transform = flipr_transform + self._xaxis_transform

        # This is the transform for r-axis ticks.  It scales the theta
        # axis so the gridlines from 0.0 to 1.0, now go from thetamin to
        # thetamax.
        self._yaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.BboxTransformTo(self.viewLim),
                mtransforms.IdentityTransform()) +
            self.transData)
        # The r-axis labels are put at an angle and padded in the r-direction
        self._r_label_position = mtransforms.Affine2D() \
            .translate(self._default_rlabel_position, 0.0)
        self._yaxis_text_transform = mtransforms.TransformWrapper(
            self._r_label_position + self.transData)

    def get_xaxis_transform(self, which='grid'):
        cbook._check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        if which in ('tick1', 'tick2'):
            return self._yaxis_text_transform
        elif which == 'grid':
            return self._yaxis_transform
        else:
            cbook._check_in_list(['tick1', 'tick2', 'grid'], which=which)

    def get_yaxis_text1_transform(self, pad):
        thetamin, thetamax = self._realViewLim.intervalx
        if _is_full_circle_rad(thetamin, thetamax):
            return self._yaxis_text_transform, 'bottom', 'left'
        elif self.get_theta_direction() > 0:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        else:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def get_yaxis_text2_transform(self, pad):
        if self.get_theta_direction() > 0:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        else:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def draw(self, *args, **kwargs):
        thetamin, thetamax = np.rad2deg(self._realViewLim.intervalx)
        if thetamin > thetamax:
            thetamin, thetamax = thetamax, thetamin
        rmin, rmax = ((self._realViewLim.intervaly - self.get_rorigin()) *
                        self.get_rsign())
        if isinstance(self.patch, mpatches.Wedge):
            # Backwards-compatibility: Any subclassed Axes might override the
            # patch to not be the Wedge that PolarAxes uses.
            center = self.transWedge.transform_point((0.5, 0.5))
            self.patch.set_center(center)
            self.patch.set_theta1(thetamin)
            self.patch.set_theta2(thetamax)

            edge, _ = self.transWedge.transform_point((1, 0))
            radius = edge - center[0]
            width = min(radius * (rmax - rmin) / rmax, radius)
            self.patch.set_radius(radius)
            self.patch.set_width(width)

            inner_width = radius - width
            inner = self.spines.get('inner', None)
            if inner:
                inner.set_visible(inner_width != 0.0)

        visible = not _is_full_circle_deg(thetamin, thetamax)
        # For backwards compatibility, any subclassed Axes might override the
        # spines to not include start/end that PolarAxes uses.
        start = self.spines.get('start', None)
        end = self.spines.get('end', None)
        if start:
            start.set_visible(visible)
        if end:
            end.set_visible(visible)
        if visible:
            yaxis_text_transform = self._yaxis_transform
        else:
            yaxis_text_transform = self._r_label_position + self.transData
        if self._yaxis_text_transform != yaxis_text_transform:
            self._yaxis_text_transform.set(yaxis_text_transform)
            self.yaxis.reset_ticks()
            self.yaxis.set_clip_path(self.patch)

        Axes.draw(self, *args, **kwargs)

    def _gen_axes_patch(self):
        return mpatches.Wedge((0.5, 0.5), 0.5, 0.0, 360.0)

    def _gen_axes_spines(self):
        spines = OrderedDict([
            ('polar', mspines.Spine.arc_spine(self, 'top',
                                              (0.5, 0.5), 0.5, 0.0, 360.0)),
            ('start', mspines.Spine.linear_spine(self, 'left')),
            ('end', mspines.Spine.linear_spine(self, 'right')),
            ('inner', mspines.Spine.arc_spine(self, 'bottom',
                                              (0.5, 0.5), 0.0, 0.0, 360.0))
        ])
        spines['polar'].set_transform(self.transWedge + self.transAxes)
        spines['inner'].set_transform(self.transWedge + self.transAxes)
        spines['start'].set_transform(self._yaxis_transform)
        spines['end'].set_transform(self._yaxis_transform)
        return spines

    def set_thetamax(self, thetamax):
        self.viewLim.x1 = np.deg2rad(thetamax)

    def get_thetamax(self):
        return np.rad2deg(self.viewLim.xmax)

    def set_thetamin(self, thetamin):
        self.viewLim.x0 = np.deg2rad(thetamin)

    def get_thetamin(self):
        return np.rad2deg(self.viewLim.xmin)

    def set_thetalim(self, *args, **kwargs):
        """
        Set the minimum and maximum theta values.

        Parameters
        ----------
        thetamin : float
            Minimum value in degrees.
        thetamax : float
            Maximum value in degrees.
        """
        if 'thetamin' in kwargs:
            kwargs['xmin'] = np.deg2rad(kwargs.pop('thetamin'))
        if 'thetamax' in kwargs:
            kwargs['xmax'] = np.deg2rad(kwargs.pop('thetamax'))
        return tuple(np.rad2deg(self.set_xlim(*args, **kwargs)))

    def set_theta_offset(self, offset):
        """
        Set the offset for the location of 0 in radians.
        """
        mtx = self._theta_offset.get_matrix()
        mtx[0, 2] = offset
        self._theta_offset.invalidate()

    def get_theta_offset(self):
        """
        Get the offset for the location of 0 in radians.
        """
        return self._theta_offset.get_matrix()[0, 2]

    def set_theta_zero_location(self, loc, offset=0.0):
        """
        Sets the location of theta's zero.  (Calls set_theta_offset
        with the correct value in radians under the hood.)

        loc : str
            May be one of "N", "NW", "W", "SW", "S", "SE", "E", or "NE".

        offset : float, optional
            An offset in degrees to apply from the specified `loc`. **Note:**
            this offset is *always* applied counter-clockwise regardless of
            the direction setting.
        """
        mapping = {
            'N': np.pi * 0.5,
            'NW': np.pi * 0.75,
            'W': np.pi,
            'SW': np.pi * 1.25,
            'S': np.pi * 1.5,
            'SE': np.pi * 1.75,
            'E': 0,
            'NE': np.pi * 0.25}
        return self.set_theta_offset(mapping[loc] + np.deg2rad(offset))

    def set_theta_direction(self, direction):
        """
        Set the direction in which theta increases.

        clockwise, -1:
           Theta increases in the clockwise direction

        counterclockwise, anticlockwise, 1:
           Theta increases in the counterclockwise direction
        """
        mtx = self._direction.get_matrix()
        if direction in ('clockwise', -1):
            mtx[0, 0] = -1
        elif direction in ('counterclockwise', 'anticlockwise', 1):
            mtx[0, 0] = 1
        else:
            cbook._check_in_list(
                [-1, 1, 'clockwise', 'counterclockwise', 'anticlockwise'],
                direction=direction)
        self._direction.invalidate()

    def get_theta_direction(self):
        """
        Get the direction in which theta increases.

        -1:
           Theta increases in the clockwise direction

        1:
           Theta increases in the counterclockwise direction
        """
        return self._direction.get_matrix()[0, 0]

    def set_rmax(self, rmax):
        """
        Set the outer radial limit.

        Parameters
        ----------
        rmax : float
        """
        self.viewLim.y1 = rmax

    def get_rmax(self):
        """
        Returns
        -------
        float
            Outer radial limit.
        """
        return self.viewLim.ymax

    def set_rmin(self, rmin):
        """
        Set the inner radial limit.

        Parameters
        ----------
        rmin : float
        """
        self.viewLim.y0 = rmin

    def get_rmin(self):
        """
        Returns
        -------
        float
            The inner radial limit.
        """
        return self.viewLim.ymin

    def set_rorigin(self, rorigin):
        """
        Update the radial origin.

        Parameters
        ----------
        rorigin : float
        """
        self._originViewLim.locked_y0 = rorigin

    def get_rorigin(self):
        """
        Returns
        -------
        float
        """
        return self._originViewLim.y0

    def get_rsign(self):
        return np.sign(self._originViewLim.y1 - self._originViewLim.y0)

    def set_rlim(self, bottom=None, top=None, emit=True, auto=False, **kwargs):
        """
        See `~.polar.PolarAxes.set_ylim`.
        """
        if 'rmin' in kwargs:
            if bottom is None:
                bottom = kwargs.pop('rmin')
            else:
                raise ValueError('Cannot supply both positional "bottom"'
                                 'argument and kwarg "rmin"')
        if 'rmax' in kwargs:
            if top is None:
                top = kwargs.pop('rmax')
            else:
                raise ValueError('Cannot supply both positional "top"'
                                 'argument and kwarg "rmax"')
        return self.set_ylim(bottom=bottom, top=top, emit=emit, auto=auto,
                             **kwargs)

    def set_ylim(self, bottom=None, top=None, emit=True, auto=False,
                 *, ymin=None, ymax=None):
        """
        Set the data limits for the radial axis.

        Parameters
        ----------
        bottom : scalar, optional
            The bottom limit (default: None, which leaves the bottom
            limit unchanged).
            The bottom and top ylims may be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

        top : scalar, optional
            The top limit (default: None, which leaves the top limit
            unchanged).

        emit : bool, optional
            Whether to notify observers of limit change (default: True).

        auto : bool or None, optional
            Whether to turn on autoscaling of the y-axis. True turns on,
            False turns off (default action), None leaves unchanged.

        ymin, ymax : scalar, optional
            These arguments are deprecated and will be removed in a future
            version.  They are equivalent to *bottom* and *top* respectively,
            and it is an error to pass both *ymin* and *bottom* or
            *ymax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new y-axis limits in data coordinates.
        """
        if ymin is not None:
            if bottom is not None:
                raise ValueError('Cannot supply both positional "bottom" '
                                 'argument and kwarg "ymin"')
            else:
                bottom = ymin
        if ymax is not None:
            if top is not None:
                raise ValueError('Cannot supply both positional "top" '
                                 'argument and kwarg "ymax"')
            else:
                top = ymax
        if top is None and np.iterable(bottom):
            bottom, top = bottom[0], bottom[1]
        return super().set_ylim(bottom=bottom, top=top, emit=emit, auto=auto)

    def get_rlabel_position(self):
        """
        Returns
        -------
        float
            The theta position of the radius labels in degrees.
        """
        return np.rad2deg(self._r_label_position.get_matrix()[0, 2])

    def set_rlabel_position(self, value):
        """Updates the theta position of the radius labels.

        Parameters
        ----------
        value : number
            The angular position of the radius labels in degrees.
        """
        self._r_label_position.clear().translate(np.deg2rad(value), 0.0)

    def set_yscale(self, *args, **kwargs):
        Axes.set_yscale(self, *args, **kwargs)
        self.yaxis.set_major_locator(
            self.RadialLocator(self.yaxis.get_major_locator(), self))

    def set_rscale(self, *args, **kwargs):
        return Axes.set_yscale(self, *args, **kwargs)

    def set_rticks(self, *args, **kwargs):
        return Axes.set_yticks(self, *args, **kwargs)

    def set_thetagrids(self, angles, labels=None, fmt=None, **kwargs):
        """
        Set the theta gridlines in a polar plot.

        Parameters
        ----------
        angles : tuple with floats, degrees
            The angles of the theta gridlines.

        labels : tuple with strings or None
            The labels to use at each theta gridline. The
            `.projections.polar.ThetaFormatter` will be used if None.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'. Note that the angle that is used is in
            radians.

        Returns
        -------
        lines, labels : list of `.lines.Line2D`, list of `.text.Text`
            *lines* are the theta gridlines and *labels* are the tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `~.Text` properties for the labels.

        See Also
        --------
        .PolarAxes.set_rgrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """

        # Make sure we take into account unitized data
        angles = self.convert_yunits(angles)
        angles = np.deg2rad(angles)
        self.set_xticks(angles)
        if labels is not None:
            self.set_xticklabels(labels)
        elif fmt is not None:
            self.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        for t in self.xaxis.get_ticklabels():
            t.update(kwargs)
        return self.xaxis.get_ticklines(), self.xaxis.get_ticklabels()

    def set_rgrids(self, radii, labels=None, angle=None, fmt=None,
                   **kwargs):
        """
        Set the radial gridlines on a polar plot.

        Parameters
        ----------
        radii : tuple with floats
            The radii for the radial gridlines

        labels : tuple with strings or None
            The labels to use at each radial gridline. The
            `matplotlib.ticker.ScalarFormatter` will be used if None.

        angle : float
            The angular position of the radius labels in degrees.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'.

        Returns
        -------
        lines, labels : list of `.lines.Line2D`, list of `.text.Text`
            *lines* are the radial gridlines and *labels* are the tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `~.Text` properties for the labels.

        See Also
        --------
        .PolarAxes.set_thetagrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """
        # Make sure we take into account unitized data
        radii = self.convert_xunits(radii)
        radii = np.asarray(radii)

        self.set_yticks(radii)
        if labels is not None:
            self.set_yticklabels(labels)
        elif fmt is not None:
            self.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        if angle is None:
            angle = self.get_rlabel_position()
        self.set_rlabel_position(angle)
        for t in self.yaxis.get_ticklabels():
            t.update(kwargs)
        return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

    def set_xscale(self, scale, *args, **kwargs):
        if scale != 'linear':
            raise NotImplementedError(
                "You can not set the xscale on a polar plot.")

    def format_coord(self, theta, r):
        """
        Return a format string formatting the coordinate using Unicode
        characters.
        """
        if theta < 0:
            theta += 2 * np.pi
        theta /= np.pi
        return ('\N{GREEK SMALL LETTER THETA}=%0.3f\N{GREEK SMALL LETTER PI} '
                '(%0.3f\N{DEGREE SIGN}), r=%0.3f') % (theta, theta * 180.0, r)

    def get_data_ratio(self):
        '''
        Return the aspect ratio of the data itself.  For a polar plot,
        this should always be 1.0
        '''
        return 1.0

    # # # Interactive panning

    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.

        Polar axes do not support zoom boxes.
        """
        return False

    def can_pan(self):
        """
        Return *True* if this axes supports the pan/zoom button functionality.

        For polar axes, this is slightly misleading. Both panning and
        zooming are performed by the same button. Panning is performed
        in azimuth while zooming is done along the radial.
        """
        return True

    def start_pan(self, x, y, button):
        angle = np.deg2rad(self.get_rlabel_position())
        mode = ''
        if button == 1:
            epsilon = np.pi / 45.0
            t, r = self.transData.inverted().transform_point((x, y))
            if angle - epsilon <= t <= angle + epsilon:
                mode = 'drag_r_labels'
        elif button == 3:
            mode = 'zoom'

        self._pan_start = types.SimpleNamespace(
            rmax=self.get_rmax(),
            trans=self.transData.frozen(),
            trans_inverse=self.transData.inverted().frozen(),
            r_label_angle=self.get_rlabel_position(),
            x=x,
            y=y,
            mode=mode)

    def end_pan(self):
        del self._pan_start

    def drag_pan(self, button, key, x, y):
        p = self._pan_start

        if p.mode == 'drag_r_labels':
            startt, startr = p.trans_inverse.transform_point((p.x, p.y))
            t, r = p.trans_inverse.transform_point((x, y))

            # Deal with theta
            dt0 = t - startt
            dt1 = startt - t
            if abs(dt1) < abs(dt0):
                dt = abs(dt1) * np.sign(dt0) * -1.0
            else:
                dt = dt0 * -1.0
            dt = (dt / np.pi) * 180.0
            self.set_rlabel_position(p.r_label_angle - dt)

            trans, vert1, horiz1 = self.get_yaxis_text1_transform(0.0)
            trans, vert2, horiz2 = self.get_yaxis_text2_transform(0.0)
            for t in self.yaxis.majorTicks + self.yaxis.minorTicks:
                t.label1.set_va(vert1)
                t.label1.set_ha(horiz1)
                t.label2.set_va(vert2)
                t.label2.set_ha(horiz2)

        elif p.mode == 'zoom':
            startt, startr = p.trans_inverse.transform_point((p.x, p.y))
            t, r = p.trans_inverse.transform_point((x, y))

            # Deal with r
            scale = r / startr
            self.set_rmax(p.rmax / scale)


# to keep things all self contained, we can put aliases to the Polar classes
# defined above. This isn't strictly necessary, but it makes some of the
# code more readable (and provides a backwards compatible Polar API)
PolarAxes.PolarTransform = PolarTransform
PolarAxes.PolarAffine = PolarAffine
PolarAxes.InvertedPolarTransform = InvertedPolarTransform
PolarAxes.ThetaFormatter = ThetaFormatter
PolarAxes.RadialLocator = RadialLocator
PolarAxes.ThetaLocator = ThetaLocator
