.. _documenting-matplotlib:

**********************
Documenting matplotlib
**********************

Getting started
===============

The documentation for matplotlib is generated from ReStructured Text
using the Sphinx_ documentation generation tool. Sphinx-0.4 or later
is required. Currently this means we need to install from the svn
repository by doing::

  svn co http://svn.python.org/projects/doctools/trunk sphinx
  cd sphinx
  python setup.py install

.. _Sphinx: http://sphinx.pocoo.org/

The documentation sources are found in the `doc/` directory in the trunk.
To build the users guide in html format, cd into `doc/users_guide` and do::

  python make.py html

or::

  ./make.py html

you can also pass a ``latex`` flag to make.py to build a pdf, or pass no
arguments to build everything.

The output produced by Sphinx can be configured by editing the `conf.py`
file located in the `doc/`.

Organization of matplotlib's documentation
==========================================

The actual ReStructured Text files are kept in `doc/users`, `doc/devel`,
`doc/api` and `doc/faq`. The main entry point is `doc/index.rst`, which pulls
in the `index.rst` file for the users guide, developers guide, api reference,
and faqs. The documentation suite is built as a single document in order to
make the most effective use of cross referencing, we want to make navigating
the Matplotlib documentation as easy as possible.

Additional files can be added to the various guides by including their base
file name (the .rst extension is not necessary) in the table of contents.
It is also possible to include other documents through the use of an include
statement, such as::

  .. include:: ../../TODO

Formatting
==========

The Sphinx website contains plenty of documentation_ concerning ReST markup and
working with Sphinx in general. Here are a few additional things to keep in mind:

* Please familiarize yourself with the Sphinx directives for `inline
  markup`_. Matplotlib's documentation makes heavy use of cross-referencing and
  other semantic markup. For example, when referring to external files, use the
  ``:file:`` directive.

* Sphinx does not support tables with column- or row-spanning cells for
  latex output. Such tables can not be used when documenting matplotlib.

* Mathematical expressions can be rendered as png images in html, and in the
  usual way by latex. For example:

  ``math:`sin(x_n^2)``` yields: :math:`sin(x_n^2)`, and::

    .. math::

      \int_{-\infty}^{\infty}\frac{e^{i\phi}}{1+x^2\frac{e^{i\phi}}{1+x^2}}

  yields:

  .. math::

    \int_{-\infty}^{\infty}\frac{e^{i\phi}}{1+x^2\frac{e^{i\phi}}{1+x^2}}

* Interactive IPython sessions can be illustrated in the documentation using
  the following directive::

    .. sourcecode:: ipython

      In [69]: lines = plot([1,2,3])

  which would yield:

  .. sourcecode:: ipython

    In [69]: lines = plot([1,2,3])

.. _documentation: http://sphinx.pocoo.org/contents.html
.. _`inline markup`: http://sphinx.pocoo.org/markup/inline.html


Figures
=======

Dynamically generated figures
-----------------------------

The top level :file:`doc` dir has a folder called :file:`pyplots` in
which you should include any pyplot plotting scripts that you want to
generate figures for the documentation.  It is not necessary to
explicitly save the figure in the script, this will be done
automatically at build time to insure that the code that is included
runs and produces the advertised figure.  Several figures will be
saved with the same basnename as the filename when the documentation
is generated (low and high res PNGs, a PDF).  Matplotlib includes a
Sphinx extension (:file:`sphinxext/plot_directive.py`) for generating
the images from the python script and including either a png copy for
html or a pdf for latex::

   .. plot:: pyplot_simple.py
      :scale: 75
      :include-source:

The ``:scale:`` directive rescales the image to some percentage of the original
size. ``:include-source:`` will present the contents of the file, marked up as
source code.

Static figures
--------------

Any figures that rely on optional system configurations should be generated
with a script residing in doc/static_figs. The resulting figures will be saved
to doc/_static, and will be named based on the name of the script, so we can
avoid unintentionally overwriting any existing figures. Please run the
:file:`doc/static_figs/make.py` file when adding additional figures, and commit
the script **and** the images to svn. Please also add a line to
the README in doc/static-figs for any additional requirements necessary to
generate a new figure. These figures are not to be generated during the
documentation build. Please use something like the following to include the
source code and the image::

  .. literalinclude:: ../mpl_examples/pylab_examples/tex_unicode_demo.py

  .. image:: ../_static/tex_unicode_demo.png



.. _referring-to-mpl-docs:

Referring to mpl documents
==========================

In the documentation, you may want to include to a document in the
matplotlib src, e.g. a license file, an image file from `mpl-data`, or an
example.  When you include these files, include them using a symbolic
link from the documentation parent directory.  This way, if we
relocate the mpl documentation directory, all of the internal pointers
to files will not have to change, just the top level symlinks.  For
example, In the top level doc directory we have symlinks pointing to
the mpl `examples` and `mpl-data`::

    home:~/mpl/doc2> ls -l mpl_*
    mpl_data -> ../lib/matplotlib/mpl-data
    mpl_examples -> ../examples


In the `users` subdirectory, if I want to refer to a file in the mpl-data
directory, I use the symlink directory.  For example, from
`customizing.rst`::

   .. literalinclude:: ../mpl_data/matplotlibrc




.. _internal-section-refs:

Internal section references
===========================

To maximize internal consistency in section labeling and references,
use hypen separated, descriptive labels for section references, eg::

    .. _howto-webapp:

and refer to it using  the standard reference syntax::

    See :ref:`howto-webapp`

Keep in mind that we may want to reorganize the contents later, so
let's avoid top level names in references like ``user`` or ``devel``
or ``faq`` unless necesssary, because for example the FAQ "what is a
backend?" could later become part of the users guide, so the label::

    .. _what-is-a-backend

is better than::

    .. _faq-backend

In addition, since underscores are widely used by Sphinx itself, let's prefer
hyphens to separate words.

.. _emacs-helpers:

Section names, etc
==================

For everything but top level chapters, please use ``Upper lower`` for
section titles, eg ``Possible hangups`` rather than ``Possible
Hangups``

Emacs helpers
=============

There is an emacs mode `rst.el
<http://docutils.sourceforge.net/tools/editors/emacs/rst.el>`_ which
automates many important ReST tasks like building and updateing
table-of-contents, and promoting or demoting section headings.  Here
is the basic ``.emacs`` configuration::

    (require 'rst)
    (setq auto-mode-alist
          (append '(("\\.txt$" . rst-mode)
                    ("\\.rst$" . rst-mode)
                    ("\\.rest$" . rst-mode)) auto-mode-alist))


Some helpful functions::

    C-c TAB - rst-toc-insert

      Insert table of contents at point

    C-c C-u - rst-toc-update

        Update the table of contents at point

    C-c C-l rst-shift-region-left

        Shift region to the left

    C-c C-r rst-shift-region-right

        Shift region to the right

