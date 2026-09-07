``BarContainer`` properties and attributes
------------------------------------------

`.BarContainer` gained a new `~.BarContainer.widths` property. It returns a
list of the widths of the individual bars in the container (the dimension
perpendicular to the bar height).

For standard bar plots (e.g. created by `.Axes.bar` or `.Axes.barh`), this
reflects the width of each bar in the plot. For grouped bar plots (e.g. created
by `.Axes.grouped_bar`), each `~.BarContainer` represents one group of bars across
categories, so `~.BarContainer.widths` returns the width of each
individual bar in that group, rather than the total width of the entire group.

Additionally, `.BarContainer` gained a new ``group_positions`` attribute, which
exposes the center positions of the bar groups if the container is part of a
grouped bar plot (e.g. created by `.Axes.grouped_bar`), or ``None`` otherwise.
