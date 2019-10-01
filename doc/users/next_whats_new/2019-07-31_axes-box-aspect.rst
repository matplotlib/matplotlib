:orphan:

Setting axes box aspect
-----------------------

It is now possible to set the aspect of an axes box directly via
`~.Axes.set_box_aspect`. The box aspect is the ratio between axes height
and axes width in physical units, independent of the data limits.
This is useful to e.g. produce a square plot, independent of the data it
contains, or to have a usual plot with the same axes dimensions next to
an image plot with fixed (data-)aspect.

For use cases check out the :doc:`Axes box aspect
</gallery/subplots_axes_and_figures/axes_box_aspect>` example.
