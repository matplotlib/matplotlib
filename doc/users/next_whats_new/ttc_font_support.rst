TTC font collection support
---------------------------

Fonts in a TrueType collection file (TTC) can now be added and used. Internally,
the embedded TTF fonts are extracted and stored in the matplotlib cache
directory. Users upgrading to this version need to rebuild the font cache for
this feature to become effective.
