Font fallback in mathtext
-------------------------

Mathematical text (``mathtext`` and ``$...$`` strings) now falls back to other
fonts when a character is missing from the selected math font, mirroring the
behavior of regular text.  Previously, characters absent from the math fonts --
such as CJK characters -- were replaced by a dummy glyph; they now render using
the configured fonts and their fallback chain (with the Last Resort font as a
final fallback), just like regular text.
