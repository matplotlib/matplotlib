Development changes
-------------------

Matplotlib now requires numpy>=1.15
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib now uses Pillow to save and read pngs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The builtin png encoder and decoder has been removed, and Pillow is now a
dependency.  Note that when reading 16-bit RGB(A) images, Pillow truncates them
to 8-bit precision, whereas the old builtin decoder kept the full precision.

The deprecated wx backend (not wxagg!) now always uses wx's builtin jpeg and
tiff support rather than relying on Pillow for writing these formats; this
behavior is consistent with wx's png output.
