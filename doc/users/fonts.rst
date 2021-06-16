All about Matplotlib and Fonts!
===============================

The story of fonts has been quite eventful throughout time. It involves
contributions of tech giants such as the likes of Adobe, Apple and Microsoft.

Types
-----
In practice, there are 3 types Matplotlib supports:

.. list-table:: Type of Fonts
   :header-rows: 1

   * - Type 1 (PDF)
     - Type 3 (PDF/PS)
     - TrueType (PDF)
   * - One of the oldest types, introduced by Adobe
     - Similar to Type 1 in terms of introduction
     - Newer than previous types, most commonly used today, introduced by Apple
   * - They use simplified PostScript
       (subset of full PostScript language)
     - However, they use full PostScript language, which allows embedding
       arbitrary code!
       (in theory, even render fractals when rasterizing!)
     - They include a virtual machine that can execute code!
   * - These fonts support font hinting
     - Do not support font hinting
     - Hinting supported (virtual machine processes the "hints")
   * - Expressed in pretty compact bytecode
     - Expressed in simple ASCII form
     - Expressed in binary code points
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
goal is to find out *which* glyphs are required for a certain piece of text,
and embed only those within the output.

Since there is almost no consistency within multiple different backends and the
types of subsetting, this is generally difficult! Luckily, Matplotlib uses a
fork of an external dependency called
`ttconv <https://github.com/sandflow/ttconv>`_, which helps in embedding and
subsetting stuff. (however, recent versions have moved away from ttconv to pure
Python)

| *Type 1 fonts are still non-subsetted* through Matplotlib.
| **Type 3 and Type 42 fonts are subsetted**, with a fair amount of exceptions and bugs for the latter.
