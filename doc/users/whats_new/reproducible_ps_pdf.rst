Reproducible PS and PDF output
------------------------------

The ``SOURCE_DATE_EPOCH`` environment variable can now be used to set
the timestamp value in the PS and PDF outputs. See
https://reproducible-builds.org/specs/source-date-epoch/

The reproducibility of the output from the PS and PDF backends has so
far been tested using various plot elements but only default values of
options such as ``{ps,pdf}.fonttype`` that can affect the output at a
low level, and not with the mathtext or usetex features. When
matplotlib calls external tools (such as PS distillers or LaTeX) their
versions need to be kept constant for reproducibility, and they may
add sources of nondeterminism outside the control of matplotlib.
