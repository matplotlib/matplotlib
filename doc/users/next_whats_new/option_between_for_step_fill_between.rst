step() and fill_between() take a new option where/step="between"
------------------------------------------------------------------------

Previously one would need to trick step() and fill_between() to plot
data where x has one point than y, typically when plotting pre-binned
histograms.

step() now takes where="between" for x, y satisfying either
len(x) + 1 = len(y) or len(x) = len(y) + 1. Plotting a step line "between"
specified edges in either direction. Convenience option where="edges" is
added to close the shape.

fill_between() now takes step="between" for x, y satisfying
len(x) + 1 = len(y). Plotting fill "between" specified edges.

fill_betweenx() now takes step="between" for x, y satisfying
len(x) = len(y) + 1. Plotting fill "between" specified edges.
