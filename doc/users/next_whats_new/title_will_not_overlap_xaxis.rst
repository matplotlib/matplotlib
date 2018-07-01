Axes title will no longer overlap xaxis
---------------------------------------

Previously an axes title had to be moved manually if an xaxis overlapped
(usually when the xaxis was put on the top of the axes).  Now, the title
will be automatically moved above the xaxis and its decorators (including
the xlabel) if they are at the top.

If desired, the title can still be placed manually.  There is a slight kludge;
the algorithm checks if the y-position of the title is 1.0 (the default),
and moves if it is.  If the user places the title in the default location
(i.e. ``ax.title.set_position(0.5, 1.0)``), the title will still be moved
above the xaxis.  If the user wants to avoid this, they can
specify a number that is close (i.e. ``ax.title.set_position(0.5, 1.01)``)
and the title will not be moved via this algorithm.
