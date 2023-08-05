Tk backend respects file format selection when saving figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When saving a figure from a Tkinter GUI to a filename without an
extension, the file format is now selected based on the value of
the dropdown menu, rather than defaulting to PNG. When the filename
contains an extension, or the OS automatically appends one, the
behavior remains unchanged.
