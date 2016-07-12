Reproducible PS and PDF output
------------------------------

The ``SOURCE_DATE_EPOCH`` environment variable can now be used to set
the timestamps value in the PS and PDF outputs. See
https://reproducible-builds.org/specs/source-date-epoch/

Matplotlib does its best to make PS and PDF outputs reproducible, but
be aware that some unreproducibility issues can arise if you use
different versions of Matplotlib and the tools it relies on. Although
standard plots has been checked to be reproducible, external tools can
also be a source of nondeterminism (``mathtext``, ``ps.usedistiller``,
``ps.fonttype``, ``pdf.fonttype``...).
