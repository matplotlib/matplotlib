Setting norms with strings
~~~~~~~~~~~~~~~~~~~~~~~~~~
Norms can now be set (e.g. on images) using the string name of the
corresponding scale, e.g. ``imshow(array, norm="log")``.  Note that in that
case, it is permissible to also pass *vmin* and *vmax*, as a new Norm instance
will be created under the hood.
