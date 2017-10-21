Axes title will no longer overlap xaxis
---------------------------------------

Previously the axes title had to be moved manually if an xaxis overlapped
(usually when the xaxis was put on the top of the axes).  The title can
still be placed manually.  However, titles that need to be moved are
at ``y=1.0``,  so manualy placing at 1.0 will be moved, so if the title
is to be placed at 1.0, it should be set to something near 1.0, like 1.001.
