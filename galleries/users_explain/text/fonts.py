r"""
.. redirect-from:: /users/fonts
.. redirect-from:: /users/explain/fonts

.. _fonts:

Fonts in Matplotlib
===================

Matplotlib needs fonts to work with its text engine, some of which are shipped
alongside the installation.  The default font is `DejaVu Sans
<https://dejavu-fonts.github.io>`_ which covers most European writing systems.
However, users can configure the default fonts, and provide their own custom
fonts.  See :ref:`Customizing text properties <text_props>` for
details and :ref:`font-nonlatin` in particular for glyphs not supported by
DejaVu Sans.

Matplotlib also provides an option to offload text rendering to a TeX engine
(``usetex=True``), see :ref:`Text rendering with LaTeX
<usetex>`.

Fonts in PDF and PostScript
---------------------------

Fonts have a long (and sometimes incompatible) history in computing, leading to
different platforms supporting different types of fonts.  In practice,
Matplotlib supports three font specifications (in addition to pdf 'core fonts',
which are explained later in the guide):

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
     - Subsetted via external module ttconv
     - Subsetted via external module
       `fontTools <https://github.com/fonttools/fonttools>`__

.. note::

   Adobe disabled__ support for authoring with Type 1 fonts in January 2023.

   __ https://helpx.adobe.com/fonts/kb/postscript-type-1-fonts-end-of-support.html

Other font specifications which Matplotlib supports:

- Type 42 fonts (PS):

  - PostScript wrapper around TrueType fonts
  - 42 is the `Answer to Life, the Universe, and Everything!
    <https://en.wikipedia.org/wiki/Answer_to_Life,_the_Universe,_and_Everything>`_
  - Matplotlib uses the external library
    `fontTools <https://github.com/fonttools/fonttools>`__ to subset these types of
    fonts

- OpenType fonts:

  - OpenType is a new standard for digital type fonts, developed jointly by
    Adobe and Microsoft
  - Generally contain a much larger character set!
  - Limited support with Matplotlib

Font subsetting
^^^^^^^^^^^^^^^

The PDF and PostScript formats support embedding fonts in files, allowing the
display program to correctly render the text, independent of what fonts are
installed on the viewer's computer and without the need to pre-rasterize the text.
This ensures that if the output is zoomed or resized the text does not become
pixelated.  However, embedding full fonts in the file can lead to large output
files, particularly with fonts with many glyphs such as those that support CJK
(Chinese/Japanese/Korean).

The solution to this problem is to subset the fonts used in the document and
only embed the glyphs actually used.  This gets both vector text and small
files sizes.  Computing the subset of the font required and writing the new
(reduced) font are both complex problem and thus Matplotlib relies on
`fontTools <https://fonttools.readthedocs.io/en/latest/>`__ and a vendored fork
of ttconv.

Currently Type 3, Type 42, and TrueType fonts are subsetted.  Type 1 fonts are not.

Core Fonts
^^^^^^^^^^

In addition to the ability to embed fonts, as part of the `PostScript
<https://en.wikipedia.org/wiki/PostScript_fonts#Core_Font_Set>`_ and `PDF
specification
<https://docs.oracle.com/cd/E96927_01/TSG/FAQ/What%20are%20the%2014%20base%20fonts%20distributed%20with%20Acroba.html>`_
there are 14 Core Fonts that compliant viewers must ensure are available.  If
you restrict your document to only these fonts you do not have to embed any
font information in the document but still get vector text.

This is especially helpful to generate *really lightweight* documents::

    # trigger core fonts for PDF backend
    plt.rcParams["pdf.use14corefonts"] = True
    # trigger core fonts for PS backend
    plt.rcParams["ps.useafm"] = True

    chars = "AFM ftw!"
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, chars)

    fig.savefig("AFM_PDF.pdf", format="pdf")
    fig.savefig("AFM_PS.ps", format="ps")

Fonts in SVG
------------

Text can output to SVG in two ways controlled by :rc:`svg.fonttype`:

- as a path (``'path'``) in the SVG
- as string in the SVG with font styling on the element (``'none'``)

When saving via ``'path'`` Matplotlib will compute the path of the glyphs used
as vector paths and write those to the output.  The advantage of doing so is
that the SVG will look the same on all computers independent of what fonts are
installed.  However the text will not be editable after the fact.
In contrast, saving with ``'none'`` will result in smaller files and the
text will appear directly in the markup.  However, the appearance may vary
based on the SVG viewer and what fonts are available.

Fonts in Agg
------------

To output text to raster formats via Agg, Matplotlib relies on `FreeType
<https://www.freetype.org/>`_.  Because the exact rendering of the glyphs
changes between FreeType versions we pin to a specific version for our image
comparison tests.

How Matplotlib selects fonts
----------------------------

Internally, using a font in Matplotlib is a three step process:

1. a `.FontProperties` object is created (explicitly or implicitly)
2. based on the `.FontProperties` object the methods on `.FontManager` are used
   to select the closest "best" font Matplotlib is aware of (except for
   ``'none'`` mode of SVG).
3. the Python proxy for the font object is used by the backend code to render
   the text -- the exact details depend on the backend via `.font_manager.get_font`.

The algorithm to select the "best" font is a modified version of the algorithm
specified by the `CSS1 Specifications
<http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ which is used by web browsers.
This algorithm takes into account the font family name (e.g. "Arial", "Noto
Sans CJK", "Hack", ...), the size, style, and weight.  In addition to family
names that map directly to fonts there are five "generic font family names"
(serif, monospace, fantasy, cursive, and sans-serif) that will internally be
mapped to any one of a set of fonts.

Currently the public API for doing step 2 is `.FontManager.findfont` (and that
method on the global `.FontManager` instance is aliased at the module level as
`.font_manager.findfont`), which will only find a single font and return the absolute
path to the font on the filesystem.

Font fallback
-------------

There is no font that covers the entire Unicode space thus it is possible for the
users to require a mix of glyphs that cannot be satisfied from a single font.
While it has been possible to use multiple fonts within a Figure, on distinct
`.Text` instances, it was not previous possible to use multiple fonts in the
same `.Text` instance (as a web browser does).  As of Matplotlib 3.6 the Agg,
SVG, PDF, and PS backends will "fallback" through multiple fonts in a single
`.Text` instance:

.. plot::
   :include-source:
   :caption: The string "There are 几个汉字 in between!" rendered with 2 fonts.

   fig, ax = plt.subplots()
   ax.text(
       .5, .5, "There are 几个汉字 in between!",
       family=['DejaVu Sans', 'Noto Sans CJK JP', 'Noto Sans TC'],
       ha='center'
   )

Internally this is implemented by setting The "font family" on
`.FontProperties` objects to a list of font families.  A (currently)
private API extracts a list of paths to all of the fonts found and then
constructs a single `.ft2font.FT2Font` object that is aware of all of the fonts.
Each glyph of the string is rendered using the first font in the list that
contains that glyph.

A majority of this work was done by Aitik Gupta supported by Google Summer of
Code 2021.
"""
