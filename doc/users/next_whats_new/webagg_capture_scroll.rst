WebAgg scroll capture control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The WebAgg backend now provides the ability to capture scroll events to prevent
page scrolling when interacting with plots. This can be enabled or disabled via
the new `.FigureCanvasWebAggCore.set_capture_scroll` and
`.FigureCanvasWebAggCore.get_capture_scroll` methods.
