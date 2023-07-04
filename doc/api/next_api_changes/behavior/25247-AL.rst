``ContourSet`` is now a single Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prior to this release, `.ContourSet` (the object returned by `~.Axes.contour`)
was a custom object holding multiple `.Collection`\s (and not an `.Artist`)
-- one collection per level, each connected component of that level's contour
being an entry in the corresponding collection.

`.ContourSet` is now instead a plain `.Collection` (and thus an `.Artist`).
The collection contains a single path per contour level; this path may be
non-continuous in case there are multiple connected components.

Setting properties on the ContourSet can now usually be done using standard
collection setters (``cset.set_linewidth(3)`` to use the same linewidth
everywhere or ``cset.set_linewidth([1, 2, 3, ...])`` to set different
linewidths on each level) instead of having to go through the individual
sub-components (``cset.collections[0].set_linewidth(...)``).  Note that
during the transition period, it remains possible to access the (deprecated)
``.collections`` attribute; this causes the ContourSet to modify itself to use
the old-style multi-Collection representation.
