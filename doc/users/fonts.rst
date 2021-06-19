All about Matplotlib and Fonts!
===============================

The story of fonts has been quite eventful throughout time. It involves
contributions of tech giants such as the likes of Adobe, Apple and Microsoft.

Types
-----
In practice, there are 3 types Matplotlib supports (in addition to
'core fonts', more about which is explained later in the guide):

.. list-table:: Type of Fonts
   :header-rows: 1

   * - Type 1 (PDF)
     - Type 3 (PDF/PS)
     - TrueType (PDF)
   * - One of the oldest types, introduced by Adobe
     - Similar to Type 1 in terms of introduction
     - Newer than previous types, used commonly today, introduced by Apple
   * - Restricted subset of PostScript, charstrings are in bytecode
     - Full PostScript language, allows embedding arbitrary code
       (in theory, even render fractals when rasterizing!)
     - Include a virtual machine that can execute code!
   * - These fonts support font hinting
     - Do not support font hinting
     - Hinting supported (virtual machine processes the "hints")
   * - Difficult to subset!
     - Easy to subset!
     - Very hard to subset!

NOTE: Adobe will disable support for authoring with Type 1 fonts in
January 2023. `Read more here. <https://helpx.adobe.com/fonts/kb/postscript-type-1-fonts-end-of-support.html>`_

Special Mentions
~~~~~~~~~~~~~~~~
- Type 42 fonts (PS):

  - PostScript wrapper around TrueType fonts
  - 42 is the `Answer to Life, the Universe, and Everything! <https://en.wikipedia.org/wiki/Answer_to_Life,_the_Universe,_and_Everything>`_
  - Very hard to subset!

- OpenType fonts:

  - OpenType is a new standard for digital type fonts, developed jointly by
    Adobe and Microsoft
  - Generally contain a much larger character set!
  - Limited Support with Matplotlib


Subsetting
----------
Matplotlib is able to generate documents in multiple different formats. Some of
those formats (for example, PDF, PS/EPS, SVG) allow embedding font data in such
a way that when these documents are visually scaled, the text does not appear
pixelated.

This can be achieved by virtually embedding the *whole* font file within the
output document. However, this can lead to **very large documents**, wherein
most of the size bandwidth is captured by that font file data.

Font Subsetting is a way to embed only the *required* glyphs within the
documents. Fonts can be considered as a collection of glyphs, so ultimately the
goal is to find out *which* glyphs are required for a certain array of
characters, and embed only those within the output.

.. note::
  The role of subsetter really shines when we encounter characters like `ä`
  (composed by calling subprograms for ``a`` and ``¨``); since the subsetter
  has to find out *all* such subprograms being called by every glyph included
  in the subset, and since there is almost no consistency within multiple
  different backends and the types of subsetting, this is a generally difficult
  problem!

Luckily, Matplotlib uses a fork of an external dependency called
`ttconv <https://github.com/sandflow/ttconv>`_, which helps in embedding and
subsetting stuff. (however, recent versions have moved away from ttconv to pure
Python for certain types)

| *Type 1 fonts are still non-subsetted* through Matplotlib. (though one will only encounter these via `usetex`/`dviread` in PDF backend)
| **Type 3 and Type 42 fonts are subsetted**, with a fair amount of exceptions and bugs for the latter.

What to use?
------------
Practically, most fonts that are readily available on most operating systems or
are readily available on the internet to download include *TrueType fonts* and
its "extensions" such as MacOS-resource fork fonts and the newer OpenType
fonts.

PS and PDF backends provide support for yet another type of fonts, which remove
the need of subsetting altogether! These are called **Core Fonts**, and
Matplotlib calls them via the keyword **AFM**; all that is supplied from
Matplotlib to such documents are font metrics (specified in AFM format), and it
is the job of the viewer applications to supply the glyph definitions.

This is especially helpful to generate *really lightweight* documents.::

    # trigger core fonts for PDF backend
    plt.rcParams["pdf.use14corefonts"] = True
    # trigger core fonts for PS backend
    plt.rcParams["ps.useafm"] = True

    chars = "AFM ftw!"
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, chars)

    fig.savefig("AFM_PDF.pdf", format="pdf")
    fig.savefig("AFM_PS.ps", format="ps)

.. note::
  These core fonts are limited to PDF and PS backends only; they can not be
  rendered in other backends.
