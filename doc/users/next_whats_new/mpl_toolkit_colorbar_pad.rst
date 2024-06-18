Fix padding of single colorbar for ``ImageGrid``
------------------------------------------------

``ImageGrid`` with ``cbar_mode="single"`` no longer adds the ``axes_pad`` between the
axes and the colorbar for thr ``cbar_location`` left and bottom. Add required space
using `cbar_pad` instead.
