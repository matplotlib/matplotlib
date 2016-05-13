Improvements for the Qt figure options editor
---------------------------------------------

Various usability improvements were implemented for the Qt figure options
editor, among which:
- Line style entries are now sorted without duplicates.
- The colormap and normalization limits can now be set for images.
- Line edits for floating values now display only as many digits as necessary
  to avoid precision loss.  An important bug was also fixed regarding input
  validation using Qt5 and a locale where the decimal separator is ",".
- The axes selector now uses shorter, more user-friendly names for axes, and
  does not crash if there are no axes.
- Line and image entries using the default labels ("_lineX", "_imageX") are now
  sorted numerically even when there are more than 10 entries.
