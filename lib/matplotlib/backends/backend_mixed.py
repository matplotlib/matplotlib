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
        self._dpi = dpi

        assert not vector_renderer.option_image_nocomposite()
        self._vector_renderer = vector_renderer
        vector_renderer.start_rasterizing = self.start_rasterizing
        vector_renderer.stop_rasterizing = self.stop_rasterizing

        self._raster_renderer = None
        self._rasterizing = False

        self._renderer = self._vector_renderer

    def start_rasterizing(self):
        """
        Enter "raster" mode.  All subsequent drawing commands (until
        stop_rasterizing is called) will be drawn with the raster
        backend.

        If start_rasterizing is called multiple times before
        stop_rasterizing is called, this method has no effect.
        """
        if not self._rasterizing:
            self._raster_renderer = self._raster_renderer_class(
                self._width*self._dpi, self._height*self._dpi, self._dpi)
            self._raster_renderer.start_rasterizing = self.start_rasterizing
            self._raster_renderer.stop_rasterizing = self.stop_rasterizing
            self._renderer = self._raster_renderer
            self._rasterizing = True
        
    def stop_rasterizing(self):
        """
        Exit "raster" mode.  All of the drawing that was done since
        the last start_rasterizing command will be copied to the
        vector backend by calling draw_image.

        If stop_rasterizing is called multiple times before
        start_rasterizing is called, this method has no effect.
        """
        if self._rasterizing:
            width, height = self._width * self._dpi, self._height * self._dpi
            buffer = self._raster_renderer.buffer_rgba(0, 0)
            image = frombuffer(buffer, width, height, True)
            image.is_grayscale = False

            self._renderer = self._vector_renderer
            self._renderer.draw_image(0, 0, image, None)
            self._raster_renderer = None
            self._rasterizing = False

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self._width, self._height
        
    # The rest of this methods simply delegate to the currently active
    # rendering backend.
            
    def open_group(self, *args, **kwargs):
        return self._renderer.open_group(*args, **kwargs)
        
    def close_group(self, *args, **kwargs):
        return self._renderer.close_group(*args, **kwargs)

    def draw_path(self, *args, **kwargs):
        return self._renderer.draw_path(*args, **kwargs)

    def draw_markers(self, *args, **kwargs):
        return self._renderer.draw_markers(*args, **kwargs)

    def draw_path_collection(self, *args, **kwargs):
        return self._renderer.draw_path_collection(*args, **kwargs)

    def draw_quad_mesh(self, *args, **kwargs):
        return self._renderer.draw_quad_mesh(*args, **kwargs)

    def get_image_magnification(self, *args, **kwargs):
        return self._renderer.get_image_magnification(*args, **kwargs)

    def draw_image(self, *args, **kwargs):
        return self._renderer.draw_image(*args, **kwargs)

    def draw_tex(self, *args, **kwargs):
        return self._renderer.draw_tex(*args, **kwargs)

    def draw_text(self, *args, **kwargs):
        return self._renderer.draw_text(*args, **kwargs)

    def flipy(self, *args, **kwargs):
        return self._renderer.flipy(*args, **kwargs)
        
    def option_image_nocomposite(self, *args, **kwargs):
        return self._vector_renderer.option_image_nocomposite(*args, **kwargs)

    def get_texmanager(self, *args, **kwargs):
        return self._renderer.get_texmanager(*args, **kwargs)

    def get_text_width_height_descent(self, *args, **kwargs):
        return self._renderer.get_text_width_height_descent(*args, **kwargs)

    def new_gc(self, *args, **kwargs):
        return self._renderer.new_gc(*args, **kwargs)

    def points_to_pixels(self, *args, **kwargs):
        return self._renderer.points_to_pixels(*args, **kwargs)

    def strip_math(self, *args, **kwargs):
        return self._renderer(*args, **kwargs)
    
    def finalize(self, *args, **kwargs):
        return self._renderer.finalize(*args, **kwargs)

