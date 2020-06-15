Saving PDF metadata via PGF now consistent with PDF backend
-----------------------------------------------------------

When saving PDF files using the PGF backend, passed metadata will be
interpreted in the same way as with the PDF backend.  Previously, this metadata
was only accepted by the PGF backend when saving a multi-page PDF with
`.backend_pgf.PdfPages`, but is now allowed when saving a single figure, as
well.
