Reproducible PS, PDF and SVG output
-----------------------------------

The ``SOURCE_DATE_EPOCH`` environment variable can now be used to set
the timestamp value in the PS and PDF outputs. See
https://reproducible-builds.org/specs/source-date-epoch/

Alternatively, calling ``savefig`` with ``metadata={'creationDate': None}``
will omit the timestamp altogether.

The reproducibility of the output from the PS and PDF backends has so
far been tested using various plot elements but only default values of
options such as ``{ps,pdf}.fonttype`` that can affect the output at a
low level, and not with the mathtext or usetex features. When
matplotlib calls external tools (such as PS distillers or LaTeX) their
versions need to be kept constant for reproducibility, and they may
add sources of nondeterminism outside the control of matplotlib.

For SVG output, the ``svg.hashsalt`` rc parameter has been added in an
earlier release. This parameter changes some random identifiers in the
SVG file to be deterministic. The downside of this setting is that if
more than one file is generated using deterministic identifiers
and they end up as parts of one larger document, the identifiers can
collide and cause the different parts to affect each other.

These features are now enabled in the tests for the PDF and SVG
backends, so most test output files (but not all of them) are now
deterministic.
