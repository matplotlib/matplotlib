import numpy as np

from matplotlib import cbook
from matplotlib._tight_bbox import process_figure_for_rasterizing
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.transforms import Bbox, Affine2D, IdentityTransform


class MixedModeRenderer:
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """
    def __init__(self, figure, width, height, dpi, vector_renderer,
                 raster_renderer_class=None,
                 bbox_inches_restore=None):
        """
        Parameters
        ----------
        figure : `matplotlib.figure.Figure`
            The figure instance.
        width : scalar
            The width of the canvas in logical units
        height : scalar
            The height of the canvas in logical units
        dpi : float
            The dpi of the canvas
        vector_renderer : `matplotlib.backend_bases.RendererBase`
            An instance of a subclass of
            `~matplotlib.backend_bases.RendererBase` that will be used for the
            vector drawing.
        raster_renderer_class : `matplotlib.backend_bases.RendererBase`
            The renderer class to use for the raster drawing.  If not provided,
            this will use the Agg backend (which is currently the only viable
            option anyway.)

        """
        if raster_renderer_class is None:
            raster_renderer_class = RendererAgg

        self._raster_renderer_class = raster_renderer_class
        self._width = width
        self._height = height
        self.dpi = dpi

        self._vector_renderer = vector_renderer

        self._raster_renderer = None

        # A reference to the figure is needed as we need to change
        # the figure dpi before and after the rasterization. Although
        # this looks ugly, I couldn't find a better solution. -JJL
        self.figure = figure
        self._figdpi = figure.dpi

        self._bbox_inches_restore = bbox_inches_restore

        self._renderer = vector_renderer

    def __getattr__(self, attr):
        # Proxy everything that hasn't been overridden to the base
        # renderer. Things that *are* overridden can call methods
        # on self._renderer directly, but must not cache/store
        # methods (because things like RendererAgg change their
        # methods on the fly in order to optimise proxying down
        # to the underlying C implementation).
        return getattr(self._renderer, attr)

    # need to wrap each drawing function that might be called on the rasterized
    # version of the renderer to save what the "true" bbox is for scaling the
    # output correctly
    # the functions we might want to overwrite are:
    # `draw_path`, `draw_image`, `draw_gouraud_triangle`, `draw_text`,
    # `draw_markers`, `draw_path_collection`, `draw_quad_mesh`

    def _update_true_bbox(self, bbox, transform=None):
        """Convert to real units and update"""
        if transform is None:
            transform = IdentityTransform()
        bbox = bbox.transformed(transform + Affine2D().scale(
            self._figdpi / self.dpi))
        if self._true_bbox is None:
            self._true_bbox = bbox
        else:
            self._true_bbox = Bbox.union([self._true_bbox, bbox])

    def draw_path(self, gc, path, transform, rgbFace=None):
        if self._rasterizing > 0:
            bbox = Bbox.null()
            bbox.update_from_path(path, ignore=True)
            self._update_true_bbox(bbox, transform)
        return self._renderer.draw_path(gc, path, transform, rgbFace)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        if self._rasterizing > 0:
            bbox = Bbox.null()
            # TODO probably faster to merge all coordinates from path using
            # numpy for large lists of paths, such as the one produced by the
            # test case tests/test_backed_pgf.py:test_mixed_mode
            for path in paths:
                bbox.update_from_path(path, ignore=False)
            self._update_true_bbox(bbox, master_transform)
        return self._renderer.draw_path_collection(
                gc, master_transform, paths, all_transforms, offsets,
                offsetTrans, facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position)

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        if self._rasterizing > 0:
            # TODO should check if this is always Bbox.unit for efficiency
            bbox = Bbox.null()
            cshape = coordinates.shape
            flat_coords = coordinates.reshape((cshape[0]*cshape[1], cshape[2]))
            bbox.update_from_data_xy(flat_coords, ignore=True)
            self._update_true_bbox(bbox, master_transform)

        return self._renderer.draw_quad_mesh(
                gc, master_transform, meshWidth, meshHeight, coordinates,
                offsets, offsetTrans, facecolors, antialiased, edgecolors)

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        if self._rasterizing > 0:
            bbox = Bbox.null()
            bbox.update_from_data_xy(points, ignore=True)
            self._update_true_bbox(bbox, transform)
        return self._renderer.draw_gouraud_triangle(
                gc, points, colors, transform)

    def start_rasterizing(self):
        """
        Enter "raster" mode.  All subsequent drawing commands (until
        `stop_rasterizing` is called) will be drawn with the raster backend.
        """
        # change the dpi of the figure temporarily.
        self.figure.dpi = self.dpi
        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore)
            self._bbox_inches_restore = r

        self._raster_renderer = self._raster_renderer_class(
            self._width*self.dpi, self._height*self.dpi, self.dpi)
        self._renderer = self._raster_renderer
        self._true_bbox = None

    def stop_rasterizing(self):
        """
        Exit "raster" mode.  All of the drawing that was done since
        the last `start_rasterizing` call will be copied to the
        vector backend by calling draw_image.
        """

        self._renderer = self._vector_renderer
        height = self._height * self.dpi
        # these bounds are in pixels, relative to the figure when pixelated
        # at the requested DPI. However, the vectorized backends draw at a
        # fixed DPI of 72, and typically aren't snapped to the
        # requested-DPI pixel grid, so we have to grab the actual bounds to
        # put the image into some other way
        if self._true_bbox is not None:
            #    raise NotImplementedError(
            #        "Something was drawn using a method not wrapped by "
            #        "MixedModeRenderer.")
            img = np.asarray(self._raster_renderer.buffer_rgba())
            slice_y, slice_x = cbook._get_nonzero_slices(img[..., 3])
            cropped_img = img[slice_y, slice_x]
            if cropped_img.size:
                gc = self._renderer.new_gc()
                # TODO: If the mixedmode resolution differs from the figure's
                #       dpi, the image must be scaled (dpi->_figdpi). Not all
                #       backends support this.
                # because rasterizing will have rounded size to nearest
                # pixel, we need to rescale our drawing to fit the original
                # intended Bbox. This results in a slightly different DPI than
                # requested, but that's better than the drawing not fitting
                # into the space requested, see Issue #6827

                self._renderer.draw_image(
                    gc, self._true_bbox.x0, self._true_bbox.y0, cropped_img[::-1],
                    true_size=(self._true_bbox.width, self._true_bbox.height)
                )

        self._raster_renderer = None

        # restore the figure dpi.
        self.figure.dpi = self._figdpi

        if self._bbox_inches_restore:  # when tight bbox is used
            r = process_figure_for_rasterizing(self.figure,
                                               self._bbox_inches_restore,
                                               self._figdpi)
            self._bbox_inches_restore = r
