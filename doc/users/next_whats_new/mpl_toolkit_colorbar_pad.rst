Fix padding of single colorbar for ``ImageGrid``
------------------------------------------------

``ImageGrid`` with ``cbar_mode="single"`` no longer adds the ``axes_pad`` between the
axes and the colorbar for ``cbar_location`` "left" and "bottom". If desired, add additional spacing
using ``cbar_pad``.
