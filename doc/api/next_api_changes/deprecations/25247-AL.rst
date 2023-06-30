``ContourSet.collections``
~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  `.ContourSet` is now implemented as a single `.Collection` of paths,
each path corresponding to a contour level, possibly including multiple unconnected
components.

During the deprecation period, accessing ``ContourSet.collections`` will revert the
current ContourSet instance to the old object layout, with a separate `.PathCollection`
per contour level.
