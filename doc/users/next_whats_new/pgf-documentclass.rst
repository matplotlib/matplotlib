PGF backend: new rcParam :rc:`pgf.documentclass`
------------------------------------------------

A new rcParam :rc:`pgf.documentclass` has been added to allow users to override
the default LaTeX document class (``article``) used by the PGF backend.
This enables better compatibility when including PGF figures in documents that
use custom LaTeX classes like ``IEEEtran`` or others, avoiding layout
issues like incorrect font sizes or spacing mismatches.
