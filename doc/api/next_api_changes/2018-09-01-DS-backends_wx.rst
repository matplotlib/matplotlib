wx Backends
-----------

The internal implementation of the wx backends was changed to do all
the screen painting inside the ``_OnPaint`` method which handles wx
``EVT_PAINT`` events.
So for a screen update due to a call to ``draw`` or for drawing a
selection rubberband, the ``Refresh`` method is called to trigger
a later paint event instead of drawing directly to a ``ClientDC``.

The atribute ``_retinaFix`` has moved from ``NavigationToolbar2Wx``
to ``_FigureCanvasWxBase``.

The method ``gui_repaint`` of all wx canvases has been removed.
The ``draw`` method no longer accepts an argument ``drawDC``.
