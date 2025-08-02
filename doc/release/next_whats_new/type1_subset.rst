PDF files created with usetex now embed subsets of Type 1 fonts
---------------------------------------------------------------

When using the PDF backend with the usetex feature,
Matplotlib calls TeX to render the text and formulas in the figure.
The fonts that get used are usually "Type 1" fonts.
They used to be embedded in full
but are now limited to the glyphs that are actually used in the figure.
This reduces the size of the resulting PDF files.
