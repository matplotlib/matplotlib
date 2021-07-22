Type 1 fonts are now subsetted in PDF output
--------------------------------------------

When using the usetex feature with the PDF backend, Type 1 fonts are embedded
in the PDF output. These fonts used to be embedded in full, but they are now
subsetted to only include the glyphs that are actually used in the figure.
