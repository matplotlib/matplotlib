Support for loading TrueType Collection fonts
---------------------------------------------

TrueType Collection fonts (commonly found as files with a ``.ttc`` extension) are now
supported. Namely, Matplotlib will include these file extensions in its scan for system
fonts, and will add all sub-fonts to its list of available fonts (i.e., the list from
`~.font_manager.get_font_names`).

From most high-level API, this means you should be able to specify the name of any
sub-font in a collection just as you would any other font. Note that at this time, there
is no way to specify the entire collection with any sort of automated selection of the
internal sub-fonts.

In the low-level API, to ensure backwards-compatibility while facilitating this new
support, a `.FontPath` instance (comprised of a font path and a sub-font index, with
behaviour similar to a `str`) may be passed to the font management API in place of a
simple `os.PathLike` path. Any font management API that previously returned a string path
now returns a `.FontPath` instance instead.
