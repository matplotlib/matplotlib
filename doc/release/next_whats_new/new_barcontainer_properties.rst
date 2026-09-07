``BarContainer`` properties and attributes
------------------------------------------

In addition to the existing `~.BarContainer.bottoms`, `~.BarContainer.tops`,
and `~.BarContainer.position_centers` properties, `.BarContainer` gained a new
`~.BarContainer.widths` property.

Additionally, `.BarContainer` gained a new ``group_positions`` attribute, which
exposes the center positions of the bar groups if the container is part of a
grouped bar plot (e.g. created by `.Axes.grouped_bar`), or ``None`` otherwise.

