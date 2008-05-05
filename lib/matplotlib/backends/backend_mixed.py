from matplotlib._image import frombuffer
from matplotlib.backends.backend_agg import RendererAgg

class MixedModeRenderer(object):
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """
    def __init__(self, width, height, dpi, vector_renderer, raster_renderer_class=None):
        """
        width: The width of the canvas in logical units

        height: The height of the canvas in logical units

        dpi: The dpi of the canvas

        vector_renderer: An instance of a subclass of RendererBase
        that will be used for the vector drawing.

        raster_renderer_class: The renderer class to use for the
        raster drawing.  If not provided, this will use the Agg
        backend (which is currently the only viable option anyway.)
        """
        if raster_renderer_class is None:
            raster_renderer_class = RendererAgg

        self._raster_renderer_class = raster_renderer_class
        self._width = width
        self._height = height
        self.dpi = dpi

        assert not vector_renderer.option_image_nocomposite()
        self._vector_renderer = vector_renderer

        self._raster_renderer = None
        self._rasterizing = 0

        self._set_current_renderer(vector_renderer)

    _methods = """
        close_group draw_image draw_markers draw_path
        draw_path_collection draw_quad_mesh draw_tex draw_text
        finalize flipy get_canvas_width_height get_image_magnification
        get_texmanager get_text_width_height_descent new_gc open_group
        option_image_nocomposite points_to_pixels strip_math
        """.split()
    def _set_current_renderer(self, renderer):
        self._renderer = renderer

        for method in self._methods:
            if hasattr(renderer, method):
                setattr(self, method, getattr(renderer, method))
        renderer.start_rasterizing = self.start_rasterizing
        renderer.stop_rasterizing = self.stop_rasterizing

    def start_rasterizing(self):
        """
        Enter "raster" mode.  All subsequent drawing commands (until
        stop_rasterizing is called) will be drawn with the raster
        backend.

        If start_rasterizing is called multiple times before
        stop_rasterizing is called, this method has no effect.
        """
        if self._rasterizing == 0:
            self._raster_renderer = self._raster_renderer_class(
                self._width*self.dpi, self._height*self.dpi, self.dpi)
            self._set_current_renderer(self._raster_renderer)
        self._rasterizing += 1

    def stop_rasterizing(self):
        """
        Exit "raster" mode.  All of the drawing that was done since
        the last start_rasterizing command will be copied to the
        vector backend by calling draw_image.

        If stop_rasterizing is called multiple times before
        start_rasterizing is called, this method has no effect.
        """
        self._rasterizing -= 1
        if self._rasterizing == 0:
            self._set_current_renderer(self._vector_renderer)

            width, height = self._width * self.dpi, self._height * self.dpi
            buffer, bounds = self._raster_renderer.tostring_rgba_minimized()
            l, b, w, h = bounds
            if w > 0 and h > 0:
                image = frombuffer(buffer, w, h, True)
                image.is_grayscale = False
                image.flipud_out()
                self._renderer.draw_image(l, height - b - h, image, None)
            self._raster_renderer = None
            self._rasterizing = False
