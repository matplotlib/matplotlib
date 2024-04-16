r"""

.. redirect-from:: /tutorials/text/pgf

.. _pgf:

************************************************************
Text rendering with XeLaTeX/LuaLaTeX via the ``pgf`` backend
************************************************************

Using the ``pgf`` backend, Matplotlib can export figures as pgf drawing
commands that can be processed with pdflatex, xelatex or lualatex. XeLaTeX and
LuaLaTeX have full Unicode support and can use any font that is installed in
the operating system, making use of advanced typographic features of OpenType,
AAT and Graphite. Pgf pictures created by ``plt.savefig('figure.pgf')``
can be embedded as raw commands in LaTeX documents. Figures can also be
directly compiled and saved to PDF with ``plt.savefig('figure.pdf')`` by
switching the backend ::

    matplotlib.use('pgf')

or by explicitly requesting the use of the ``pgf`` backend ::

    plt.savefig('figure.pdf', backend='pgf')

or by registering it for handling pdf output ::

    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

The last method allows you to keep using regular interactive backends and to
save xelatex, lualatex or pdflatex compiled PDF files from the graphical user
interface.  Note that, in that case, the interactive display will still use the
standard interactive backends (e.g., QtAgg), and in particular use latex to
compile relevant text snippets.

Matplotlib's pgf support requires a recent LaTeX_ installation that includes
the TikZ/PGF packages (such as TeXLive_), preferably with XeLaTeX or LuaLaTeX
installed. If either pdftocairo or ghostscript is present on your system,
figures can optionally be saved to PNG images as well. The executables
for all applications must be located on your :envvar:`PATH`.

`.rcParams` that control the behavior of the pgf backend:

=================  =====================================================
Parameter          Documentation
=================  =====================================================
pgf.preamble       Lines to be included in the LaTeX preamble
pgf.rcfonts        Setup fonts from rc params using the fontspec package
pgf.texsystem      Either "xelatex" (default), "lualatex" or "pdflatex"
=================  =====================================================

.. note::

   TeX defines a set of special characters, such as::

     # $ % & ~ _ ^ \ { }

   Generally, these characters must be escaped correctly. For convenience,
   some characters (_, ^, %) are automatically escaped outside of math
   environments. Other characters are not escaped as they are commonly needed
   in actual TeX expressions. However, one can configure TeX to treat them as
   "normal" characters (known as "catcode 12" to TeX) via a custom preamble,
   such as::

     plt.rcParams["pgf.preamble"] = (
         r"\AtBeginDocument{\catcode`\&=12\catcode`\#=12}")

.. _pgf-rcfonts:


Multi-Page PDF Files
====================

The pgf backend also supports multipage pdf files using
`~.backend_pgf.PdfPages`

.. code-block:: python

    from matplotlib.backends.backend_pgf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages('multipage.pdf', metadata={'author': 'Me'}) as pdf:

        fig1, ax1 = plt.subplots()
        ax1.plot([1, 5, 3])
        pdf.savefig(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot([1, 5, 3])
        pdf.savefig(fig2)


Font specification
==================

The fonts used for obtaining the size of text elements or when compiling
figures to PDF are usually defined in the `.rcParams`. You can also use the
LaTeX default Computer Modern fonts by clearing the lists for :rc:`font.serif`,
:rc:`font.sans-serif` or :rc:`font.monospace`. Please note that the glyph
coverage of these fonts is very limited. If you want to keep the Computer
Modern font face but require extended Unicode support, consider installing the
`Computer Modern Unicode`__ fonts *CMU Serif*, *CMU Sans Serif*, etc.

__ https://sourceforge.net/projects/cm-unicode/

When saving to ``.pgf``, the font configuration Matplotlib used for the
layout of the figure is included in the header of the text file.

.. literalinclude:: /gallery/userdemo/pgf_fonts.py
   :end-before: fig.savefig


.. _pgf-preamble:

Custom preamble
===============

Full customization is possible by adding your own commands to the preamble.
Use :rc:`pgf.preamble` if you want to configure the math fonts,
using ``unicode-math`` for example, or for loading additional packages. Also,
if you want to do the font configuration yourself instead of using the fonts
specified in the rc parameters, make sure to disable :rc:`pgf.rcfonts`.

.. only:: html

    .. literalinclude:: /gallery/userdemo/pgf_preamble_sgskip.py
        :end-before: fig.savefig

.. only:: latex

    .. literalinclude:: /gallery/userdemo/pgf_preamble_sgskip.py
        :end-before: import matplotlib.pyplot as plt


.. _pgf-texsystem:

Choosing the TeX system
=======================

The TeX system to be used by Matplotlib is chosen by :rc:`pgf.texsystem`.
Possible values are ``'xelatex'`` (default), ``'lualatex'`` and ``'pdflatex'``.
Please note that when selecting pdflatex, the fonts and Unicode handling must
be configured in the preamble.

.. literalinclude:: /gallery/userdemo/pgf_texsystem.py
   :end-before: fig.savefig


.. _pgf-troubleshooting:

Troubleshooting
===============

* Please note that the TeX packages found in some Linux distributions and
  MiKTeX installations are dramatically outdated. Make sure to update your
  package catalog and upgrade or install a recent TeX distribution.

* On Windows, the :envvar:`PATH` environment variable may need to be modified
  to include the directories containing the latex, dvipng and ghostscript
  executables. See :ref:`environment-variables` and
  :ref:`setting-windows-environment-variables` for details.

* Sometimes the font rendering in figures that are saved to png images is
  very bad. This happens when the pdftocairo tool is not available and
  ghostscript is used for the pdf to png conversion.

* Make sure what you are trying to do is possible in a LaTeX document,
  that your LaTeX syntax is valid and that you are using raw strings
  if necessary to avoid unintended escape sequences.

* :rc:`pgf.preamble` provides lots of flexibility, and lots of
  ways to cause problems. When experiencing problems, try to minimalize or
  disable the custom preamble.

* Configuring an ``unicode-math`` environment can be a bit tricky. The
  TeXLive distribution for example provides a set of math fonts which are
  usually not installed system-wide. XeTeX, unlike LuaLatex, cannot find
  these fonts by their name, which is why you might have to specify
  ``\setmathfont{xits-math.otf}`` instead of ``\setmathfont{XITS Math}`` or
  alternatively make the fonts available to your OS. See this
  `tex.stackexchange.com question`__ for more details.

  __ https://tex.stackexchange.com/q/43642/

* If the font configuration used by Matplotlib differs from the font setting
  in yout LaTeX document, the alignment of text elements in imported figures
  may be off. Check the header of your ``.pgf`` file if you are unsure about
  the fonts Matplotlib used for the layout.

* Vector images and hence ``.pgf`` files can become bloated if there are a lot
  of objects in the graph. This can be the case for image processing or very
  big scatter graphs.  In an extreme case this can cause TeX to run out of
  memory: "TeX capacity exceeded, sorry"  You can configure latex to increase
  the amount of memory available to generate the ``.pdf`` image as discussed on
  `tex.stackexchange.com <https://tex.stackexchange.com/q/7953/>`_.
  Another way would be to "rasterize" parts of the graph causing problems
  using either the ``rasterized=True`` keyword, or ``.set_rasterized(True)`` as
  per :doc:`this example </gallery/misc/rasterization_demo>`.

* Various math fonts are compiled and rendered only if corresponding font
  packages are loaded. Specifically, when using ``\mathbf{}`` on Greek letters,
  the default computer modern font may not contain them, in which case the
  letter is not rendered. In such scenarios, the ``lmodern`` package should be
  loaded.

* If you still need help, please see :ref:`reporting-problems`

.. _LaTeX: http://www.tug.org
.. _TeXLive: http://www.tug.org/texlive/
"""
