Change in default font
----------------------

The default font used by matplotlib in text has been changed to DejaVu Sans and
DejaVu Serif for the sans-serif and serif families, respectively. The DejaVu
font family is based on the previous matplotlib default --Bitstream Vera-- but
includes a much wider range of characters.

The default mathtext font has been changed from Computer Modern to the DejaVu
family to maintain consistency with regular text. Two new options for the
``mathtext.fontset`` configuration parameter have been added: ``dejavusans``
(default) and ``dejavuserif``. Both of these options use DejaVu glyphs whenever
possible and fall back to STIX symbols when a glyph is not found in DejaVu. To
return to the previous behavior, set the rcParam ``mathtext.fontset`` to ``cm``.
