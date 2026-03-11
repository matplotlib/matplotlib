3D annotations follow the view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~mpl_toolkits.mplot3d.axes3d.Axes3D.annotate` now accepts 3D data coordinates:
when ``xycoords='data'``, the annotated position *xy* may be passed as a
3-tuple ``(x, y, z)``.  In that case, the annotation is projected during
draws, so it stays attached to the intended point when rotating or zooming the
3D view.

The new keyword-only parameter ``axlim_clip`` can be used to hide annotations
whose 3D anchor is outside the axes view limits.

