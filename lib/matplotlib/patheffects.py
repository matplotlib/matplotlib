"""
Defines classes for path effects. The path effects are supported in
:class:`~matplotlib.text.Text`, :class:`~matplotlib.lines.Line2D`
and :class:`~matplotlib.patches.Patch`.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
import matplotlib.transforms as mtransforms
import matplotlib.cbook as cbook
from matplotlib.colors import colorConverter


class AbstractPathEffect(object):
    """
    A base class for path effects.

    Subclasses should override the ``draw_path`` method to add effect
    functionality.

    """
    def get_proxy_renderer(self, renderer):
        return ProxyRenderer(self, renderer)

    def _update_gc(self, gc, new_gc_dict):
        new_gc_dict = new_gc_dict.copy()

        dashes = new_gc_dict.pop("dashes", None)
        if dashes:
            gc.set_dashes(**dashes)

        for k, v in six.iteritems(new_gc_dict):
            set_method = getattr(gc, 'set_' + k, None)
            if set_method is None or not six.callable(set_method):
                raise AttributeError('Unknown property {}'.format(k))
            set_method(v)

        return gc

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """
        Derived should override this method. The argument is same
        as *draw_path* method of :class:`matplotlib.backend_bases.RendererBase`
        except the first argument is a renderer. The base definition is::

          def draw_path(self, renderer, gc, tpath, affine, rgbFace):
              renderer.draw_path(gc, tpath, affine, rgbFace)

        """
        renderer.draw_path(gc, tpath, affine, rgbFace)

    def draw_path_collection(self, renderer,
                             gc, master_transform, paths, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        path_ids = []
        for path, transform in renderer._iter_collection_raw_paths(
            master_transform, paths, all_transforms):
            path_ids.append((path, transform))

        for xo, yo, path_id, gc0, rgbFace in renderer._iter_collection(
            gc, master_transform, all_transforms, path_ids, offsets,
            offsetTrans, facecolors, edgecolors, linewidths, linestyles,
            antialiaseds, urls, offset_position):
            path, transform = path_id
            transform = mtransforms.Affine2D(transform.get_matrix()).translate(xo, yo)
            self.draw_path(renderer, gc0, path, transform, rgbFace)

    def draw_tex(self, renderer, gc, x, y, s, prop, angle, ismath='TeX'):
        self._draw_text_as_path(renderer, gc, x, y, s, prop, angle,
                                ismath=ismath)

    def draw_text(self, renderer, gc, x, y, s, prop, angle, ismath=False):
        self._draw_text_as_path(renderer, gc, x, y, s, prop, angle, ismath)

    def _draw_text_as_path(self, renderer, gc, x, y, s, prop, angle, ismath):
        path, transform = renderer._get_text_path_transform(x, y, s, prop,
                                                            angle, ismath)
        color = gc.get_rgb()[:3]
        gc.set_linewidth(0.0)
        self.draw_path(renderer, gc, path, transform, rgbFace=color)

    def draw_markers(self, renderer, *args, **kwargs):
        # Call the naive draw markers method which falls back to calling
        # draw_path.
        return RendererBase.draw_markers(renderer, *args, **kwargs)


class ProxyRenderer(object, RendererBase):
    def __init__(self, path_effect, renderer):
        self._path_effect = path_effect
        self._renderer = renderer

    def draw_path(self, gc, tpath, affine, rgbFace=None):
        self._path_effect.draw_path(self._renderer, gc, tpath, affine, rgbFace)

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!'):
        self._path_effect._draw_text_as_path(self._renderer,
                                             gc, x, y, s, prop, angle, ismath="TeX")

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        self._path_effect._draw_text(self.renderer,
                                     gc, x, y, s, prop, angle, ismath)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        self._path_effect.draw_markers(self._renderer,
                                       gc, marker_path, marker_trans, path, trans,
                                       rgbFace=rgbFace)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        pe = self._path_effect
        pe.draw_path_collection(self._renderer,
                                gc, master_transform, paths, all_transforms,
                                offsets, offsetTrans, facecolors, edgecolors,
                                linewidths, linestyles, antialiaseds, urls,
                                offset_position)

    def __getattr__(self, name):
        # Return the original renderer's attributes, if it hasn't been
        # overridden here.
        return getattr(self._renderer, name)


class Normal(AbstractPathEffect):
    """
    The "identity" PathEffect.

    The Normal PathEffect's sole purpose is to draw the original artist with
    no special path effect.
    """
    pass


class Stroke(AbstractPathEffect):
    """
    Stroke the path with updated gc.
    """
    def __init__(self, **kwargs):
        """
        The path will be stroked with its gc updated with the given
        keyword arguments, i.e., the keyword arguments should be valid
        gc parameter values.
        """
        super(Stroke, self).__init__()
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        draw the path with updated gc.
        """
        # Do not modify the input! Use copy instead.

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine, rgbFace)
        gc0.restore()


class withStroke(Stroke):
    """
    Adds a simple :class:`Stroke` and then draws the
    original Artist to avoid needing to call :class:`Normal`.

    """
    def draw_path(self, renderer, gc, tpath, affine, rgbFace):

        Stroke.draw_path(self, renderer, gc, tpath, affine, rgbFace)
        renderer.draw_path(gc, tpath, affine, rgbFace)


class SimplePatchShadow(AbstractPathEffect):
    """
    A simple shadow filled patch path effect.

    .. note::

        For a simple line based drop shadow ...
    """

    def __init__(self, offset_xy=(2,-2),
                 shadow_rgbFace=None, alpha=None, patch_alpha=None,
                 **kwargs):
        """
        Parameters
        ----------
        offset_xy : pair of floats
            The offset of the shadow in points.
        shadow_rgbFace : :ref:`color <mpl-color-spec>`
            The shadow color.
        alpha : float
            The alpha transparency of the created shadow patch.
            Default is 0.3.
            http://matplotlib.1069221.n5.nabble.com/path-effects-question-td27630.html
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        """
        super(AbstractPathEffect, self).__init__()
        self._offset_xy = offset_xy
        if shadow_rgbFace is None:
            self._shadow_rgbFace = shadow_rgbFace
        else:
            self._shadow_rgbFace = colorConverter.to_rgba(shadow_rgbFace)
        if patch_alpha is not None:
            cbook.deprecated('v1.4', 'The patch_alpha keyword is deprecated. '
                             'Use the alpha keyword instead. Transform your '
                             'patch_alpha by alpha = 1 - patch_alpha')
            if alpha is not None:
                raise ValueError("Both alpha and patch_alpha were set. "
                                 "Just use alpha.")
            alpha = 1 - patch_alpha

        if alpha is None:
            alpha = 0.3

        self._alpha = alpha

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

        #: The offset transform object. The offset isn't calculated yet
        #: as we don't know how big the figure will be in pixels.
        self._offset_tran = mtransforms.Affine2D()

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Overrides the standard draw_path to add the shadow offset and
        necessary color changes for the shadow.

        """
        # IMPORTANT: Do not modify the input - we copy everything instead.

        offset_x = renderer.points_to_pixels(self._offset_xy[0])
        offset_y = renderer.points_to_pixels(self._offset_xy[1])

        affine0 = affine + self._offset_tran.clear().translate(offset_x,
                                                               offset_y)

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        if self._shadow_rgbFace is None:
            r,g,b = (rgbFace or (1., 1., 1.))[:3]
            # Scale the colors by a factor to improve the shadow effect.
            rho = 0.3
            shadow_rgbFace = (r * rho, g * rho, b * rho)
        else:
            shadow_rgbFace = self._shadow_rgbFace

        gc0.set_foreground("none")
        gc0.set_alpha(self._alpha)
        gc0.set_linewidth(0)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine0, shadow_rgbFace)
        gc0.restore()


class withSimplePatchShadow(SimplePatchShadow):
    """
    Adds a simple :class:`SimplePatchShadow` and then draws the
    original Artist to avoid needing to call :class:`Normal`.

    """
    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        SimplePatchShadow.draw_path(self, renderer, gc, tpath, affine, rgbFace)
        renderer.draw_path(gc, tpath, affine, rgbFace)
