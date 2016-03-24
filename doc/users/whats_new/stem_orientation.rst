Added ``orientation`` parameter for stem plots
-----------------------------------
When creating stem plots, you can now pass in an ``orientation`` argument to :func:`stem`.

Currently, only ``vertical`` and ``horizontal`` orientations are supported,
with ``horizontal`` being the default.

Example
```````
stem(x, x, orientation='vertical')
