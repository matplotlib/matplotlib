Implementation of axes.prop_cycle support for the markevery property
--------------------------------------------------------------------

The ``rcParams`` attribute of a matplotlib session or script now supports assignments
of the attribute `axes.prop_cycle` composed from cyclers including the `markevery` property.
The full API of the `~matplotlib.lines.set_markevery` method is enabled 
via the new `validate_markevery` method of rcsetup.py. A demonstration is available at 
`~matplotlib/examples/lines_bars_and_markers/markevery_propcycles_demo.py`
