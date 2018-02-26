Deprecations
````````````
The following functions and classes are deprecated:

- ``cbook.GetRealpathAndStat`` (which is only a helper for
  ``get_realpath_and_stat``),
- ``cbook.is_numlike`` (use ``isinstance(..., numbers.Number)`` instead),
- ``FigureCanvasWx.frame`` and ``.tb`` (use ``.window`` and ``.toolbar``
  respectively, which is consistent with other backends),
- the ``FigureCanvasWx`` constructor should not be called with ``(parent, id,
  figure)`` as arguments anymore, but just ``figure`` (like all other canvas
  classes).  Call ``Reparent`` and ``SetId`` to set the parent and id of the
  canvas.
- ``mathtext.unichr_safe`` (use ``chr`` instead),
