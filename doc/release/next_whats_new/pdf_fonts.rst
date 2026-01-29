Improved font embedding in PDF
------------------------------

Both Type 3 and Type 42 fonts (see :ref:`fonts` for more details) are now
embedded into PDFs without limitation. Fonts may be split into multiple
embedded subsets in order to satisfy format limits. Additionally, a corrected
Unicode mapping is added for each.

This means that *all* text should now be selectable and copyable in PDF viewers
that support doing so.
