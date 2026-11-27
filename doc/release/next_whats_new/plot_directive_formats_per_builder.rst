Per-builder formats for the plot directive
------------------------------------------

The ``plot_formats`` setting of `matplotlib.sphinxext.plot_directive` can now
be a dict that gives each Sphinx builder its own list of formats, keyed by
builder name or builder format, the same names that the
:external+sphinx:rst:dir:`only` directive accepts::

    plot_formats = {'html': ['png', 'hires.png'], 'latex': ['pdf']}

A PDF added for the LaTeX build then no longer shows up as a download link on
the HTML pages. Builders that are not in the dict use the default list, and an
entry that lists a format its builder cannot use produces a warning. A
single entry can also be set on the command line. For example,
``-D plot_formats.html=png`` gives the HTML build only a PNG, and with the
formats separated by commas, ``-D plot_formats.html=png:100,pdf`` gives it a
100 dpi PNG and a PDF.
