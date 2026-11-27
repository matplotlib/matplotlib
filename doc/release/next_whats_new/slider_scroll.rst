Opt-in scrolling for sliders
----------------------------

Pass ``scroll=True`` to `~matplotlib.widgets.Slider` to let users adjust
its value by scrolling over it. Each scroll event moves one step in the
scroll direction. The step follows *valstep* when provided and is 1% of
the slider's range otherwise.

See :doc:`/gallery/widgets/slider_demo` for an example.
