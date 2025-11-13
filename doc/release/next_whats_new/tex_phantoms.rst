mathtext support for ``\phantom``, ``\llap``, ``\rlap``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mathtext gained support for the TeX macros ``\phantom``, ``\llap``, and
``\rlap``. ``\phantom`` allows to occupy some space on the canvas as if
some text was being rendered, without actually rendering that text, whereas
``\llap`` and ``\rlap`` allows to render some text on the canvas while
pretending that it occupies no space.  Altogether these macros allow some finer
control of text alignments.

See https://www.tug.org/TUGboat/tb22-4/tb72perlS.pdf for a detailed description
of these macros.
