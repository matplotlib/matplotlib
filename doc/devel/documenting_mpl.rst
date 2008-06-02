.. _documenting-matplotlib:

**********************
Documenting Matplotlib
**********************

The documentation for matplotlib is generated from ReStructured Text
using the Sphinx_ documentation generation tool. Sphinx-0.4 or later
is required. Currently this means we need to install from the svn
repository by doing::

  svn co http://svn.python.org/projects/doctools/trunk sphinx
  cd sphinx
  python setup.py install

.. _Sphinx: http://sphinx.pocoo.org/

The documentation sources are found in the doc/ directory in the trunk.
To build the users guid in html format, cd into doc/users_guide and do::

  python make.py html

you can also pass a ``latex`` flag to make.py to build a pdf, or pass no
arguments to build everything. The same procedure can be followed for
the sources in doc/api_reference.

The actual ReStructured Text files are kept in doc/users_guide/source
and doc/api_reference/source. The main entry point is index.rst.
Additional files can be added by including their base file name
(dropping the .rst extension) in the table of contents. It is also
possible to include other documents through the use of an include
statement. For example, in the Developers Guide, index.rst lists
coding_guide, which automatically inserts coding_guide.rst.

Sphinx does not support tables with column- or row-spanning cells for
latex output. Such tables can not be used when documenting matplotlib.

Mathematical expressions can be rendered as png images in html, and in
the usual way by latex. For example:

``math:`sin(x_n^2)`` yields: :math:`sin(x_n^2)`, and::

  .. math::

     \int_{-\infty}^{\infty}\frac{e^{i\phi}}{1+x^2\frac{e^{i\phi}}{1+x^2}}``

yields:

.. math::

   \int_{-\infty}^{\infty}\frac{e^{i\phi}}{1+x^2\frac{e^{i\phi}}{1+x^2}}

The output produced by Sphinx can be configured by editing the conf.py
files located in the documentation source directories.

The Sphinx website contains plenty of documentation_ concerning ReST
markup and working with Sphinx in general.

.. _documentation: http://sphinx.pocoo.org/contents.html

.. _referring-to-mpl-docs:

Referring to mpl documents
==========================

In the documentation, you may want to include to a document in the
matplotlib src, eg a license file, an image file from ``mpl-data``, or an
example.  When you include these files, include them using a symbolic
link from the documentation parent directory.  This way, if we
relocate the mpl documentation directory, all of the internal pointers
to files will not have to change, just the top level symlinks.  For
example, In the top level doc directory we have symlinks pointing to
the mpl ``examples`` and ``mpl-data``::

    home:~/mpl/doc2> ls -l mpl_*
    mpl_data -> ../lib/matplotlib/mpl-data
    mpl_examples -> ../examples


In the ``users`` subdirectory, if I want to refer to a file in the mpl-data directory, I use the symlink direcotry.  For example, from ``customizing.rst``::

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

In addition, since underscores are widely used by Sphinx itself, let's prefer hyphens to separate words.

.. _emacs-helpers:

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

