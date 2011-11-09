"""
Defines classes for path effects. The path effects are supported in
:class:`~matplotlib.text.Text` and :class:`~matplotlib.patches.Patch`
matplotlib.text.Text.
"""

from __future__ import print_function
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
import matplotlib.transforms as transforms



class _Base(object):
    """
    A base class for PathEffect. Derived must override draw_path method.
    """

    def __init__(self):
        """
        initializtion.
        """
        super(_Base, self).__init__()


    def _update_gc(self, gc, new_gc_dict):
        new_gc_dict = new_gc_dict.copy()

        dashes = new_gc_dict.pop("dashes", None)
        if dashes:
            gc.set_dashes(**dashes)

        for k, v in new_gc_dict.iteritems():
            set_method = getattr(gc, 'set_'+k, None)
            if set_method is None or not callable(set_method):
                raise AttributeError('Unknown property %s'%k)
            set_method(v)

        return gc


    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Derived should override this method. The argument is same
        as *draw_path* method of :class:`matplotlib.backend_bases.RendererBase`
        except the first argument is a renderer. The base definition is ::

          def draw_path(self, renderer, gc, tpath, affine, rgbFace):
                  renderer.draw_path(gc, tpath, affine, rgbFace)

        """
        renderer.draw_path(gc, tpath, affine, rgbFace)

    def draw_tex(self, renderer, gc, x, y, s, prop, angle, ismath='TeX!'):
        self._draw_text_as_path(renderer, gc, x, y, s, prop, angle, ismath="TeX")

    def draw_text(self, renderer, gc, x, y, s, prop, angle, ismath=False):
        self._draw_text_as_path(renderer, gc, x, y, s, prop, angle, ismath)

    def _draw_text_as_path(self, renderer, gc, x, y, s, prop, angle, ismath):

        if isinstance(renderer, MixedModeRenderer):
            renderer = renderer._renderer

        path, transform = RendererBase._get_text_path_transform(renderer,
                                                                x, y, s,
                                                                prop, angle,
                                                                ismath)
        color = gc.get_rgb()[:3]

        gc.set_linewidth(0.0)
        self.draw_path(renderer, gc, path, transform, rgbFace=color)


#     def draw_path_collection(self, renderer,
#                              gc, master_transform, paths, all_transforms,
#                              offsets, offsetTrans, facecolors, edgecolors,
#                              linewidths, linestyles, antialiaseds, urls):
#         path_ids = []
#         for path, transform in renderer._iter_collection_raw_paths(
#             master_transform, paths, all_transforms):
#             path_ids.append((path, transform))

#         for xo, yo, path_id, gc0, rgbFace in renderer._iter_collection(
#             gc, path_ids, offsets, offsetTrans, facecolors, edgecolors,
#             linewidths, linestyles, antialiaseds, urls):
#             path, transform = path_id
#             transform = transforms.Affine2D(transform.get_matrix()).translate(xo, yo)
#             self.draw_path(renderer, gc0, path, transform, rgbFace)


class Normal(_Base):
    """
    path effect with no effect
    """
    pass

class Stroke(_Base):
    """
    stroke the path with updated gc.
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
        draw the path with update gc.
        """
        # Do not modify the input! Use copy instead.

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine, rgbFace)
        gc0.restore()


class withStroke(Stroke):

    """
    Same as Stroke, but add a stroke with the original gc at the end.
    """

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):

        Stroke.draw_path(self, renderer, gc, tpath, affine, rgbFace)
        renderer.draw_path(gc, tpath, affine, rgbFace)


import matplotlib.transforms as mtransforms

class SimplePatchShadow(_Base):
    """
    simple shadow
    """

    def __init__(self, offset_xy=(2,-2),
                 shadow_rgbFace=None, patch_alpha=0.7,
                 **kwargs):
        """
        """
        super(_Base, self).__init__()
        self._offset_xy = offset_xy
        self._shadow_rgbFace = shadow_rgbFace
        self._patch_alpha = patch_alpha

        self._gc = kwargs
        self._offset_tran = mtransforms.Affine2D()

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        """
        # Do not modify the input! Use copy instead.

        offset_x = renderer.points_to_pixels(self._offset_xy[0])
        offset_y = renderer.points_to_pixels(self._offset_xy[1])

        affine0 = affine + self._offset_tran.clear().translate(offset_x, offset_y)

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        if self._shadow_rgbFace is None:
            r,g,b = rgbFace[:3]
            rho = 0.3
            r = rho*r
            g = rho*g
            b = rho*b

            shadow_rgbFace = (r,g,b)
        else:
            shadow_rgbFace = self._shadow_rgbFace

        gc0.set_foreground("none")
        gc0.set_alpha(1.-self._patch_alpha)
        gc0.set_linewidth(0)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine0, shadow_rgbFace)
        gc0.restore()


class withSimplePatchShadow(SimplePatchShadow):
    """
    simple shadow
    """

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):

        SimplePatchShadow.draw_path(self, renderer, gc, tpath, affine, rgbFace)

        gc1 = renderer.new_gc()
        gc1.copy_properties(gc)
        gc1.set_alpha(gc1.get_alpha()*self._patch_alpha)
        renderer.draw_path(gc1, tpath, affine, rgbFace)
        gc1.restore()


if __name__ == '__main__':
    clf()
    imshow([[1,2],[2,3]])
    #eff = PathEffects.Thicken()
    txt = annotate("test", (1., 1.), (0., 0),
                   arrowprops=dict(arrowstyle="->", connectionstyle="angle3", lw=2),
                   size=12, ha="center")
    txt.set_path_effects([withStroke(linewidth=3, foreground="w")])
    #txt.arrow_patch.set_path_effects([PathEffects.withStroke(width=3, color="w")])
    txt.arrow_patch.set_path_effects([Stroke(linewidth=5, foreground="w"),
                                      Normal()])
