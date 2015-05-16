Reordered `Axes.get_children`
`````````````````````````````

The artist order returned by `Axes.get_children` did not
match the one used by `Axes.draw`.  They now use the same
order, as `Axes.draw` now calls `Axes.get_children`.
