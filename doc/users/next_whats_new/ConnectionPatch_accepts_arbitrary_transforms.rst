:orphan:

`ConnectionPatch` accepts arbitrary transforms
----------------------------------------------

Alternatively to strings like ``"data"`` or ``"axes fraction"``
`ConnectionPatch` now accepts any `~matplotlib.transforms.Transform`
as input for the ``coordsA`` and ``coordsB`` argument. This allows to
draw lines between points defined in different user defined coordinate
systems. Also see the :doc:`Connect Simple01 example
</gallery/userdemo/connect_simple01>`.