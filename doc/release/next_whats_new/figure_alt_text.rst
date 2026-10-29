Figures can carry alternative text
----------------------------------

Figures now have alternative text ("alt text"): a plain-text description
of the figure for people who cannot see it, e.g. when using a screen
reader. Set it with `.Figure.set_alt_text` and read it back with
`.Figure.get_alt_text`::

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.set_alt_text("Line plot rising from (0, 0) to (1, 1)")

When saving to SVG, the root ``<svg>`` element gets a ``role="img"``
attribute and the alt text is written as its ``<title>`` child, which
makes it the accessible name of the figure. On the other backends that
accept file metadata (PNG, PDF, PostScript, PGF), the alt text is stored
as the ``'Description'`` metadata entry instead, unless a description is
explicitly passed to `.Figure.savefig`; note that file metadata is not
an accessible name, and PDF output does not become a tagged PDF.
