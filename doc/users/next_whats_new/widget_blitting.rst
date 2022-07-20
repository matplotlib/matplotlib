Blitting in Button widgets
--------------------------

The `.Button`, `.CheckButtons`, and `.RadioButtons` widgets now support
blitting for faster rendering, on backends that support it, by passing
``useblit=True`` to the constructor. Blitting is enabled by default on
supported backends.
