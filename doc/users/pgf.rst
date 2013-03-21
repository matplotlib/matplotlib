.. _pgf-tutorial:

*********************************
Typesetting With XeLaTeX/LuaLaTeX
*********************************

Using the ``pgf`` backend, matplotlib can export figures as pgf drawing commands
that can be processed with pdflatex, xelatex or lualatex. XeLaTeX and LuaLaTeX
have full unicode support and can use any font that is installed in the operating
system, making use of advanced typographic features of OpenType, AAT and
Graphite. Pgf pictures created by ``plt.savefig('figure.pgf')`` can be
embedded as raw commands in LaTeX documents. Figures can also be directly
compiled and saved to PDF with ``plt.savefig('figure.pdf')`` by either
switching to the backend

.. code-block:: python

    matplotlib.use('pgf')

or registering it for handling pdf output

.. code-block:: python

    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

The second method allows you to keep using regular interactive backends and to
save xelatex, lualatex or pdflatex compiled PDF files from the graphical user interface.

Matplotlib's pgf support requires a recent LaTeX_ installation that includes
the TikZ/PGF packages (such as TeXLive_), preferably with XeLaTeX or LuaLaTeX
installed. If either pdftocairo or ghostscript is present on your system,
figures can optionally be saved to PNG images as well. The executables
for all applications must be located on your :envvar:`PATH`.

Rc parameters that control the behavior of the pgf backend:

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
   some characters (_,^,%) are automatically escaped outside of math
   environments.

.. _pgf-rcfonts:

Font specification
==================

The fonts used for obtaining the size of text elements or when compiling
figures to PDF are usually defined in the matplotlib rc parameters. You can
also use the LaTeX default Computer Modern fonts by clearing the lists for
``font.serif``, ``font.sans-serif`` or ``font.monospace``. Please note that
the glyph coverage of these fonts is very limited. If you want to keep the
Computer Modern font face but require extended unicode support, consider
installing the `Computer Modern Unicode <http://sourceforge.net/projects/cm-unicode/>`_
fonts *CMU Serif*, *CMU Sans Serif*, etc.

When saving to ``.pgf``, the font configuration matplotlib used for the
layout of the figure is included in the header of the text file.

.. literalinclude:: plotting/examples/pgf_fonts.py
   :end-before: plt.savefig

.. image:: /_static/pgf_fonts.*


.. _pgf-preamble:

Custom preamble
===============

Full customization is possible by adding your own commands to the preamble.
Use the ``pgf.preamble`` parameter if you want to configure the math fonts,
using ``unicode-math`` for example, or for loading additional packages. Also,
if you want to do the font configuration yourself instead of using the fonts
specified in the rc parameters, make sure to disable ``pgf.rcfonts``.

.. htmlonly::

    .. literalinclude:: plotting/examples/pgf_preamble.py
        :end-before: plt.savefig

.. latexonly::

    .. literalinclude:: plotting/examples/pgf_preamble.py
        :end-before: import matplotlib.pyplot as plt

.. image:: /_static/pgf_preamble.*


.. _pgf-texsystem:

Choosing the TeX system
=======================

The TeX system to be used by matplotlib is chosen by the ``pgf.texsystem``
parameter. Possible values are ``'xelatex'`` (default), ``'lualatex'`` and
``'pdflatex'``. Please note that when selecting pdflatex the fonts and
unicode handling must be configured in the preamble.

.. literalinclude:: plotting/examples/pgf_texsystem.py
   :end-before: plt.savefig

.. image:: /_static/pgf_texsystem.*


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

* A limitation on Windows causes the backend to keep file handles that have
  been opened by your application open. As a result, it may not be possible
  to delete the corresponding files until the application closes (see
  `#1324 <https://github.com/matplotlib/matplotlib/issues/1324>`_).

* Sometimes the font rendering in figures that are saved to png images is
  very bad. This happens when the pdftocairo tool is not available and
  ghostscript is used for the pdf to png conversion.

* Make sure what you are trying to do is possible in a LaTeX document,
  that your LaTeX syntax is valid and that you are using raw strings
  if necessary to avoid unintended escape sequences.

* The ``pgf.preamble`` rc setting provides lots of flexibility, and lots of
  ways to cause problems. When experiencing problems, try to minimalize or
  disable the custom preamble.

* Configuring an ``unicode-math`` environment can be a bit tricky. The
  TeXLive distribution for example provides a set of math fonts which are
  usually not installed system-wide. XeTeX, unlike LuaLatex, cannot find
  these fonts by their name, which is why you might have to specify
  ``\setmathfont{xits-math.otf}`` instead of ``\setmathfont{XITS Math}`` or
  alternatively make the fonts available to your OS. See this
  `tex.stackexchange.com question <http://tex.stackexchange.com/questions/43642>`_
  for more details.

* If the font configuration used by matplotlib differs from the font setting
  in yout LaTeX document, the alignment of text elements in imported figures
  may be off. Check the header of your ``.pgf`` file if you are unsure about
  the fonts matplotlib used for the layout.

* If you still need help, please see :ref:`reporting-problems`

.. _LaTeX: http://www.tug.org
.. _TeXLive: http://www.tug.org/texlive/
