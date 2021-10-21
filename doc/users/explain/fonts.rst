.. redirect-from:: /users/fonts
  
Fonts in Matplotlib text engine
===============================

Matplotlib needs fonts to work with its text engine, some of which are shipped
alongside the installation. However, users can configure the default fonts, or
even provide their own custom fonts! For more details, see :doc:`Customizing
text properties </tutorials/text/text_props>`.

However, Matplotlib also provides an option to offload text rendering to a TeX
engine (``usetex=True``),
see :doc:`Text rendering with LaTeX </tutorials/text/usetex>`.

Font specifications
-------------------
Fonts have a long and sometimes incompatible history in computing, leading to
different platforms supporting different types of fonts. In practice, there are
3 types of font specifications Matplotlib supports (in addition to 'core
fonts', more about which is explained later in the guide):

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
   * - Non-subsetted through Matplotlib
     - Subsetted via external module `ttconv <https://github.com/sandflow/ttconv>`_
     - Subsetted via external module `fonttools <https://github.com/fonttools/fonttools>`_

NOTE: Adobe will disable support for authoring with Type 1 fonts in
January 2023. `Read more here. <https://helpx.adobe.com/fonts/kb/postscript-type-1-fonts-end-of-support.html>`_

Special mentions
^^^^^^^^^^^^^^^^
Other font specifications which Matplotlib supports:

- Type 42 fonts (PS):

  - PostScript wrapper around TrueType fonts
  - 42 is the `Answer to Life, the Universe, and Everything! <https://en.wikipedia.org/wiki/Answer_to_Life,_the_Universe,_and_Everything>`_
  - Matplotlib uses an external library called `fonttools <https://github.com/fonttools/fonttools>`_
    to subset these types of fonts

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

This can be achieved by embedding the *whole* font file within the
output document. However, this can lead to very large documents, as some
fonts (for instance, CJK - Chinese/Japanese/Korean fonts) can contain a large
number of glyphs, and thus their embedded size can be quite huge.

Font Subsetting can be used before generating documents, to embed only the
*required* glyphs within the documents. Fonts can be considered as a collection
of glyphs, so ultimately the goal is to find out *which* glyphs are required
for a certain array of characters, and embed only those within the output.

.. note::
  The role of subsetter really shines when we encounter characters like **ä**
  (composed by calling subprograms for **a** and **¨**); since the subsetter
  has to find out *all* such subprograms being called by every glyph included
  in the subset, this is a generally difficult problem!

Luckily, Matplotlib uses a fork of an external dependency called
`ttconv <https://github.com/sandflow/ttconv>`_, which helps in embedding and
subsetting font data. (however, recent versions have moved away from ttconv to
pure Python for certain types: for more details visit
`these <https://github.com/matplotlib/matplotlib/pull/18370>`_, `links <https://github.com/matplotlib/matplotlib/pull/18181>`_)

| *Type 1 fonts are still non-subsetted* through Matplotlib. (though one will encounter these mostly via *usetex*/*dviread* in PDF backend)
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

  Another downside to this is that while the font metrics are standardized,
  different PDF viewer applications will have different fonts to render these
  metrics. In other words, the **output might look different on different
  viewers**, as well as (let's say) Windows and Linux, if Linux tools included
  free versions of the proprietary fonts.

  This also violates the *what-you-see-is-what-you-get* feature of Matplotlib.
