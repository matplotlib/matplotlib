====================
MEP14: Text handling
====================

.. contents::
   :local:


Status
======

- **Discussion**

Branches and Pull requests
==========================

Issue #253 demonstrates a bug where using the bounding box rather than
the advance width of text results in misaligned text.  This is a minor
point in the grand scheme of things, but it should be addressed as
part of this MEP.

Abstract
========

By reorganizing how text is handled, this MEP aims to:

- improve support for Unicode and non-ltr languages
- improve text layout (especially multi-line text)
- allow support for more fonts, especially non-Apple-format TrueType
  fonts and OpenType fonts.
- make the font configuration easier and more transparent

Detailed description
====================

**Text layout**

At present, matplotlib has two different ways to render text:
"built-in" (based on FreeType and our own Python code), and "usetex"
(based on calling out to a TeX installation).  Adjunct to the
"built-in" renderer there is also the Python-based "mathtext" system
for rendering mathematical equations using a subset of the TeX
language without having a TeX installation available.  Support for
these two engines in strewn about many source files, including every
backend, where one finds clauses like ::

  if rcParams['text.usetex']: # do one thing else: # do another

Adding a third text rendering approach (more on that later) would
require editing all of these places as well, and therefore doesn't
scale.

Instead, this MEP proposes adding a concept of "text engines", where
the user could select one of many different approaches for rendering
text.  The implementations of each of these would be localized to
their own set of modules, and not have little pieces around the whole
source tree.

Why add more text rendering engines?  The "built-in" text rendering
has a number of shortcomings.

- It only handles right-to-left languages, and doesn't handle many
  special features of Unicode, such as combining diacriticals.
- The multiline support is imperfect and only supports manual
  line-breaking -- it can not break up a paragraph into lines of a
  certain length.
- It also does not handle inline formatting changes in order to
  support something like Markdown, reStructuredText or HTML.  (Though
  rich-text formatting is contemplated in this MEP, since we want to
  make sure this design allows it, the specifics of a rich-text
  formatting implementation is outside of the scope of this MEP.)

Supporting these things is difficult, and is the "full-time job" of a
number of other projects:

  - `pango <http://www.pango.org/>`_/`harfbuzz
    <https://www.freedesktop.org/wiki/Software/HarfBuzz/>`_
  - `QtTextLayout
    <http://doc.qt.io/qt-4.8/qtextlayout.html>`_
  - `Microsoft DirectWrite
    <https://msdn.microsoft.com/en-us/library/windows/desktop/dd371554(v=vs.85).aspx>`_
  - `Apple Core Text
    <https://developer.apple.com/library/content/documentation/StringsTextFonts/Conceptual/CoreText_Programming/Overview/Overview.html>`_

Of the above options, it should be noted that `harfbuzz` is designed
from the start as a cross platform option with minimal dependencies,
so therefore is a good candidate for a single option to support.

Additionally, for supporting rich text, we could consider using
`WebKit <https://webkit.org/>`_, and possibly whether than
represents a good single cross-platform option.  Again, however, rich
text formatting is outside of the scope of this project.

Rather than trying to reinvent the wheel and add these features to
matplotlib's "built-in" text renderer, we should provide a way to
leverage these projects to get more powerful text layout.  The
"built-in" renderer will still need to exist for reasons of ease of
installation, but its feature set will be more limited compared to the
others.  [TODO: This MEP should clearly decide what those limited
features are, and fix any bugs to bring the implementation into a
state of working correctly in all cases that we want it to work.  I
know @leejjoon has some thoughts on this.]

**Font selection**

Going from an abstract description of a font to a file on disk is the
task of the font selection algorithm -- it turns out to be much more
complicated than it seems at first.

The "built-in" and "usetex" renderers have very different ways of
handling font selection, given their different technologies.  TeX
requires the installation of TeX-specific font packages, for example,
and can not use TrueType fonts directly.  Unfortunately, despite the
different semantics for font selection, the same set of font
properties are used for each.  This is true of both the
`FontProperties` class and the font-related `rcParams` (which
basically share the same code underneath).  Instead, we should define
a core set of font selection parameters that will work across all text
engines, and have engine-specific configuration to allow the user to
do engine-specific things when required.  For example, it is possible
to directly select a font by name in the "built-in" using
`font.family`, but the same is not possible with "usetex".  It may be
possible to make it easier to use TrueType fonts by using XeTeX, but
users will still want to use the traditional metafonts through TeX
font packages.  So the issue still stands that different text engines
will need engine-specific configuration, and it should be more obvious
to the user which configuration will work across text engines and
which are engine-specific.

Note that even excluding "usetex", there are different ways to find
fonts.  The default is to use the font list cache in `font_manager.py`
which matches fonts using our own algorithm based on the `CSS font
matching algorithm <http://www.w3.org/TR/CSS2/fonts.html#algorithm>`_.
It doesn't always do the same thing as the native font selection
algorithms on Linux (`fontconfig
<https://www.freedesktop.org/wiki/Software/fontconfig/>`_), Mac and
Windows, and it doesn't always find all of the fonts on the system
that the OS would normally pick up.  However, it is cross-platform,
and always finds the fonts that ship with matplotlib.  The Cairo and
MacOSX backends (and presumably a future HTML5-based backend)
currently bypass this mechanism and use the OS-native ones.  The same
is true when not embedding fonts in SVG, PS or PDF files and opening
them in a third-party viewer.  A downside there is that (at least with
Cairo, need to confirm with MacOSX) they don't always find the fonts
we ship with matplotlib.  (It may be possible to add the fonts to
their search path, though, or we may need to find a way to install our
fonts to a location the OS expects to find them).

There are also special modes in the PS and PDF to only use the core
fonts that are always available to those formats.  There, the font
lookup mechanism must only match against those fonts.  It is unclear
whether the OS-native font lookup systems can handle this case.

There is also experimental support for using `fontconfig
<https://www.freedesktop.org/wiki/Software/fontconfig/>`_ for font
selection in matplotlib, turned off by default.  fontconfig is the
native font selection algorithm on Linux, but is also cross platform
and works well on the other platforms (though obviously is an
additional dependency there).

Many of the text layout libraries proposed above (pango, QtTextLayout,
DirectWrite and CoreText etc.) insist on using the font selection
library from their own ecosystem.

All of the above seems to suggest that we should move away from our
self-written font selection algorithm and use the native APIs where
possible.  That's what Cairo and MacOSX backends already want to use,
and it will be a requirement of any complex text layout library.  On
Linux, we already have the bones of a `fontconfig` implementation
(which could also be accessed through pango).  On Windows and Mac we
may need to write custom wrappers.  The nice thing is that the API for
font lookup is relatively small, and essentially consist of "given a
dictionary of font properties, give me a matching font file".

**Font subsetting**

Font subsetting is currently handled using ttconv.  ttconv was a
standalone commandline utility for converting TrueType fonts to
subsetted Type 3 fonts (among other features) written in 1995, which
matplotlib (well, I) forked in order to make it work as a library.  It
only handles Apple-style TrueType fonts, not ones with the Microsoft
(or other vendor) encodings.  It doesn't handle OpenType fonts at all.
This means that even though the STIX fonts come as .otf files, we have
to convert them to .ttf files to ship them with matplotlib.  The Linux
packagers hate this -- they'd rather just depend on the upstream STIX
fonts.  ttconv has also been shown to have a few bugs that have been
difficult to fix over time.

Instead, we should be able to use FreeType to get the font outlines
and write our own code (probably in Python) to output subsetted fonts
(Type 3 on PS and PDF and SVGFonts or paths on SVG).  Freetype, as a
popular and well-maintained project, handles a wide variety of fonts
in the wild.  This would remove a lot of custom C code, and remove
some code duplication between backends.

Note that subsetting fonts this way, while the easiest route, does
lose the hinting in the font, so we will need to continue, as we do
now, provide a way to embed the entire font in the file where
possible.

Alternative font subsetting options include using the subsetting
built-in to Cairo (not clear if it can be used without the rest of
Cairo), or using `fontforge` (which is a heavy and not terribly
cross-platform dependency).

**Freetype wrappers**

Our FreeType wrapper could really use a reworking.  It defines its own
image buffer class (when a Numpy array would be easier).  While
FreeType can handle a huge diversity of font files, there are
limitations to our wrapper that make it much harder to support
non-Apple-vendor TrueType files, and certain features of OpenType
files.  (See #2088 for a terrible result of this, just to support the
fonts that ship with Windows 7 and 8).  I think a fresh rewrite of
this wrapper would go a long way.

**Text anchoring and alignment and rotation**

The handling of baselines was changed in 1.3.0 such that the backends
are now given the location of the baseline of the text, not the bottom
of the text.  This is probably the correct behavior, and the MEP
refactoring should also follow this convention.

In order to support alignment on multi-line text, it should be the
responsibility of the (proposed) text engine to handle text alignment.
For a given chunk of text, each engine calculates a bounding box for
that text and the offset of the anchor point within that box.
Therefore, if the va of a block was "top", the anchor point would be
at the top of the box.

Rotating of text should always be around the anchor point.  I'm not
sure that lines up with current behavior in matplotlib, but it seems
like the sanest/least surprising choice.  [This could be revisited
once we have something working].  Rotation of text should not be
handled by the text engine -- that should be handled by a layer
between the text engine and the rendering backend so it can be handled
in a uniform way.  [I don't see any advantage to rotation being
handled by the text engines individually...]

There are other problems with text alignment and anchoring that should
be resolved as part of this work.  [TODO: enumerate these].

**Other minor problems to fix**

The mathtext code has backend-specific code -- it should instead
provide its output as just another text engine.  However, it's still
desirable to have mathtext layout inserted as part of a larger layout
performed by another text engine, so it should be possible to do this.
It's an open question whether embedding the text layout of an
arbitrary text engine in another should be possible.

The text mode is currently set by a global rcParam ("text.usetex") so
it's either all on or all off.  We should continue to have a global
rcParam to choose the text engine ("text.layout_engine"), but it
should under the hood be an overridable property on the `Text` object,
so the same figure can combine the results of multiple text layout
engines if necessary.


Implementation
==============

A concept of a "text engine" will be introduced.  Each text engine
will implement a number of abstract classes.  The `TextFont` interface
will represent text for a given set of font properties.  It isn't
necessarily limited to a single font file -- if the layout engine
supports rich text, it may handle a number of font files in a family.
Given a `TextFont` instance, the user can get a `TextLayout` instance,
which represents the layout for a given string of text in a given
font.  From a `TextLayout`, an iterator over `TextSpans` is returned
so the engine can output raw editable text using as few spans as
possible.  If the engine would rather get individual characters, they
can be obtained from the `TextSpan` instance::


  class TextFont(TextFontBase):
      def __init__(self, font_properties):
          """
          Create a new object for rendering text using the given font properties.
          """
          pass

      def get_layout(self, s, ha, va):
          """
          Get the TextLayout for the given string in the given font and
          the horizontal (left, center, right) and verticalalignment (top,
          center, baseline, bottom)
          """
          pass

  class TextLayout(TextLayoutBase):
      def get_metrics(self):
          """
          Return the bounding box of the layout, anchored at (0, 0).
          """
          pass

      def get_spans(self):
          """
          Returns an iterator over the spans of different in the layout.
          This is useful for backends that want to editable raw text as
          individual lines.  For rich text where the font may change,
          each span of different font type will have its own span.
          """
          pass

      def get_image(self):
          """
          Returns a rasterized image of the text.  Useful for raster backends,
          like Agg.

          In all likelihood, this will be overridden in the backend, as it can
          be created from get_layout(), but certain backends may want to
          override it if their library provides it (as freetype does).
          """
          pass

      def get_rectangles(self):
          """
          Returns an iterator over the filled black rectangles in the layout.
          Used by TeX and mathtext for drawing, for example, fraction lines.
          """
          pass

      def get_path(self):
          """
          Returns a single Path object of the entire laid out text.

          [Not strictly necessary, but might be useful for textpath
          functionality]
          """
          pass

  class TextSpan(TextSpanBase):
      x, y      # Position of the span -- relative to the text layout as a whole
                # where (0, 0) is the anchor.  y is the baseline of the span.
      fontfile  # The font file to use for the span
      text      # The text content of the span

      def get_path(self):
          pass  # See TextLayout.get_path

      def get_chars(self):
          """
          Returns an iterator over the characters in the span.
          """
          pass

  class TextChar(TextCharBase):
      x, y      # Position of the character -- relative to the text layout as
                # a whole, where (0, 0) is the anchor.  y is in the baseline
                # of the character.
      codepoint # The unicode code point of the character -- only for informational
                # purposes, since the mapping of codepoint to glyph_id may have been
                # handled in a complex way by the layout engine.  This is an int
                # to avoid problems on narrow Unicode builds.
      glyph_id  # The index of the glyph within the font
      fontfile  # The font file to use for the char

      def get_path(self):
          """
          Get the path for the character.
          """
  pass


Graphic backends that want to output subset of fonts would likely
build up a file-global dictionary of characters where the keys are
(fontname, glyph_id) and the values are the paths so that only one
copy of the path for each character will be stored in the file.

Special casing: The "usetex" functionality currently is able to get
Postscript directly from TeX to insert directly in a Postscript file,
but for other backends, parses a DVI file and generates something more
abstract.  For a case like this, `TextLayout` would implement
`get_spans` for most backends, but add `get_ps` for the Postscript
backend, which would look for the presence of this method and use it
if available, or fall back to `get_spans`.  This kind of special
casing may also be necessary, for example, when the graphics backend
and text engine belong to the same ecosystem, e.g. Cairo and Pango, or
MacOSX and CoreText.

There are three main pieces to the implementation:

1) Rewriting the freetype wrapper, and removing ttconv.

 a) Once (1) is done, as a proof of concept, we can move to the
    upstream STIX .otf fonts

 b) Add support for web fonts loaded from a remote URL.  (Enabled by using freetype for font subsetting).

2) Refactoring the existing "builtin" and "usetex" code into separate text engines and to follow the API outlined above.

3) Implementing support for advanced text layout libraries.


(1) and (2) are fairly independent, though having (1) done first will
allow (2) to be simpler.  (3) is dependent on (1) and (2), but even if
it doesn't get done (or is postponed), completing (1) and (2) will
make it easier to move forward with improving the "builtin" text
engine.

Backward compatibility
======================

The layout of text with respect to its anchor and rotation will change
in hopefully small, but improved, ways.  The layout of multiline text
will be much better, as it will respect horizontal alignment.  The
layout of bidirectional text or other advanced Unicode features will
now work inherently, which may break some things if users are
currently using their own workarounds.

Fonts will be selected differently.  Hacks that used to sort of work
between the "builtin" and "usetex" text rendering engines may no
longer work.  Fonts found by the OS that weren't previously found by
matplotlib may be selected.

Alternatives
============

TBD
