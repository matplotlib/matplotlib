Cursor text now uses a number of significant digits matching pointing precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the x/y position displayed by the cursor text would usually include
far more significant digits than the mouse pointing precision (typically one
pixel).  This is now fixed for linear scales.
