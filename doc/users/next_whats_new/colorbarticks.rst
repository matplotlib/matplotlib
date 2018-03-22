Colorbar ticks can now be automatic
-----------------------------------

The number of ticks on colorbars was appropriate for a large colorbar, but
looked bad if the colorbar was made smaller (i.e. via the ``shrink`` kwarg).
This has been changed so that the number of ticks is now responsive to how
large the colorbar is.  
