Path simplification updates
---------------------------

Line simplification controlled by the ``path.simplify`` and
``path.simplify_threshold`` parameters has been improved. You should
notice better rendering performance when plotting large amounts of
data (as long as the above parameters are set accordingly). Only the
line segment portion of paths will be simplified -- if you are also
drawing markers and experiencing problems with rendering speed, you
should consider using the ``markevery`` option to ``plot``.
See the :ref:`performance` section in the usage tutorial for more
information.

The simplification works by iteratively merging line segments
into a single vector until the next line segment's perpendicular
distance to the vector (measured in display-coordinate space)
is greater than the ``path.simplify_threshold`` parameter. Thus, higher
values of ``path.simplify_threshold`` result in quicker rendering times.
If you are plotting just to explore data and not for publication quality,
pixel perfect plots, then a value of ``1.0`` can be safely used. If you
want to make sure your plot reflects your data *exactly*, then you should
set ``path.simplify`` to false and/or ``path.simplify_threshold`` to ``0``.
Matplotlib currently defaults to a conservative value of ``1/9``, smaller
values are unlikely to cause any visible differences in your plots.
