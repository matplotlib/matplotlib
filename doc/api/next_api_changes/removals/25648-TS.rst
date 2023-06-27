``Axes3D.set_frame_on`` and ``Axes3D.get_frame_on`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Axes3D.set_frame_on`` is documented as "Set whether the 3D axes panels are
drawn.". However, it has no effect on 3D axes and is being removed in
favor of ``Axes3D.set_axis_on`` and ``Axes3D.set_axis_off``.
