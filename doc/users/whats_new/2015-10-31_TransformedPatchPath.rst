New TransformedPatchPath caching object
---------------------------------------

A newly added :class:`~matplotlib.transforms.TransformedPatchPath` provides a
means to transform a :class:`~matplotlib.patches.Patch` into a
:class:`~matplotlib.path.Path` via a :class:`~matplotlib.transforms.Transform`
while caching the resulting path. If neither the patch nor the transform have
changed, a cached copy of the path is returned.

This class differs from the older
:class:`~matplotlib.transforms.TransformedPath` in that it is able to refresh
itself based on the underlying patch while the older class uses an immutable
path.
