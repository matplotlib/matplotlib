
API Changes in 2.1.2
====================

`Figure.legend` no longer checks for repeated lines to ignore
-------------------------------------------------------------

`matplotlib.Figure.legend` used to check if a line had the
same label as an existing legend entry. If it also had the same line color
or marker color legend didn't add a new entry for that line. However, the
list of conditions was incomplete, didn't handle RGB tuples,
didn't handle linewidths or linestyles etc.

This logic did not exist in `Axes.legend`.  It was included (erroneously)
in Matplotlib 2.1.1 when the legend argument parsing was unified
[#9324](https://github.com/matplotlib/matplotlib/pull/9324).  This change
removes that check in `Axes.legend` again to restore the old behavior.

This logic has also been dropped from `.Figure.legend`, where it
was previously undocumented. Repeated
lines with the same label will now each have an entry in the legend.  If
you do not want the duplicate entries, don't add a label to the line, or
prepend the label with an underscore.
