API Changes for 3.10.1
======================

Behaviour
---------

*alpha* parameter handling on images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When passing and array to ``imshow(..., alpha=...)``, the parameter was silently ignored
if the image data was a RGB or RBGA image or if :rc:`interpolation_state`
resolved to "rbga".

This is now fixed, and the alpha array overwrites any previous transparency information.
