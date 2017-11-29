.. _documenting-matplotlib:

=====================
Writing documentation
=====================

Getting started
===============

Installing dependencies
-----------------------

The documentation for Matplotlib is generated from reStructuredText using
the Sphinx_ documentation generation tool. There are several extra requirements
that are needed to build the documentation. They are listed in
:file:`doc-requirements.txt` and listed below:

1. Sphinx 1.3 or later (1.5.0 is not supported)
2. numpydoc 0.4 or later
3. IPython
4. mock
5. colorspacious
6. Pillow
7. Graphviz

.. note::

  * You'll need a minimal working LaTeX distribution for many examples to run.
  * `Graphviz <http://www.graphviz.org/Download.php>`_ is not a Python package,
    and needs to be installed separately.

General file structure
----------------------

All documentation is built from the :file:`doc/` directory.  This directory
contains both ``.rst`` files that contain pages in the documentation and
configuration files for Sphinx_.

The ``.rst`` files are kept in :file:`doc/users`,
:file:`doc/devel`, :file:`doc/api` and :file:`doc/faq`. The main entry point is
:file:`doc/index.rst`, which pulls in the :file:`index.rst` file for the users
guide, developers guide, api reference, and FAQs. The documentation suite is
built as a single document in order to make the most effective use of cross
referencing.

.. note::

   An exception to this are the directories :file:`examples` and
   :file:`tutorials`, which exist in the root directory.  These contain Python
   files that are built by `Sphinx Gallery`_.  When the docs are built,
   the directories :file:`docs/gallery` and :file:`docs/tutorials`
   are automatically generated. Do not edit the rst files in :file:docs/gallery
   and :file:docs/tutorials, as they are rebuilt from the original sources in
   the root directory.


Additional files can be added to the various guides by including their base
file name (the .rst extension is not necessary) in the table of contents.
It is also possible to include other documents through the use of an include
statement, such as::

  .. include:: ../../TODO

The configuration file for Sphinx is :file:`doc/conf.py`. It controls which
directories Sphinx parses, how the docs are built, and how the extensions are
used.

Building the docs
-----------------

The documentation sources are found in the :file:`doc/` directory in the trunk.
To build the documentation in html format, cd into :file:`doc/` and run:

.. code-block:: sh

   python make.py html

The list of commands and flags for ``make.py`` can be listed by running
``python make.py --help``. In particular,

* ``python make.py clean`` will delete the built Sphinx files. Use this command
  if you're getting strange errors about missing paths or broken links,
  particularly if you move files around.
* ``python make.py latex`` builds a PDF of the documentation.
* The ``--allowsphinxwarnings`` flag allows the docs to continue building even
  if Sphinx throws a warning. This is useful for debugging and spot-checking
  many warnings at once.

.. _formatting-mpl-docs:

Writing new documentation
=========================

Most documentation lives in "docstrings". These are comment blocks in source
code that explain how the code works. All new or edited docstrings should
conform to the numpydoc guidelines. These split the docstring into a number
of sections - see
https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt for more
details and a guide to how docstrings should be formatted.

An example docstring looks like:

.. code-block:: python

  def hlines(self, y, xmin, xmax, colors='k', linestyles='solid',
             label='', **kwargs):
        """
        Plot horizontal lines at each *y* from *xmin* to *xmax*.

        Parameters
        ----------
        y : scalar or sequence of scalar
            y-indexes where to plot the lines.

        xmin, xmax : scalar or 1D array_like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have same length.

        colors : array_like of colors, optional, default: 'k'

        linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

        label : string, optional, default: ''

        Returns
        -------
        lines : `~matplotlib.collections.LineCollection`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.collections.LineCollection` properties.

        See also
        --------
        vlines : vertical lines
        axhline: horizontal line across the axes
        """

The Sphinx website also contains plenty of documentation_ concerning ReST
markup and working with Sphinx in general.

.. note::

   Some parts of the documentation do not yet conform to the current
   documentation style. If in doubt, follow the rules given here and not what
   you may see in the source code. Pull requests updating docstrings to
   the current style are very welcome.

Additional formatting conventions
---------------------------------

There are some additional conventions, not handled by numpydoc and the Sphinx
guide:

* We do not have a convention whether to use single-quotes or double-quotes.
  There is a mixture of both in the current code. Please leave them as they are.

* Long parameter lists should be wrapped using a ``\`` for continuation and
  starting on the new line without any indent:

  .. code-block:: python

     def add_axes(self, *args, **kwargs):
         """
         ...

         Parameters
         ----------
         projection :
             ['aitoff' | 'hammer' | 'lambert' | 'mollweide' | \
     'polar' | 'rectilinear'], optional
             The projection type of the axes.

  Alternatively, you can describe the valid parameter values in a dedicated
  section of the docstring.

* Generally, do not add markup to types for ``Parameters`` and ``Returns``.
  This is usually not needed because Sphinx will link them automatically and
  would unnecessarily clutter the docstring. However, it does seem to fail in
  some situations. If you encounter such a case, you are allowed to add markup:

  .. code-block:: rst

     Returns
     -------
     lines : `~matplotlib.collections.LineCollection`



Linking to other code
---------------------
To link to other methods, classes, or modules in Matplotlib you can encase
the name to refer to in back ticks, for example:

.. code-block:: python

  `~matplotlib.collections.LineCollection`

It is also possible to add links to code in Python, Numpy, Scipy, or Pandas.
Sometimes it is tricky to get external Sphinx linking to work; to check that
a something exists to link to the following shell command outputs a list of all
objects that can be referenced (in this case for Numpy)::

  python -m sphinx.ext.intersphinx 'https://docs.scipy.org/doc/numpy/objects.inv'


Function arguments
------------------
Function arguments and keywords within docstrings should be referred to using
the ``*emphasis*`` role. This will keep Matplotlib's documentation consistent
with Python's documentation:

.. code-block:: rst

  Here is a description of *argument*

Please do not use the ```default role```:


.. code-block:: rst

   Please do not describe `argument` like this.

nor the ````literal```` role:

.. code-block:: rst

   Please do not describe ``argument`` like this.

Setters and getters
-------------------
Matplotlib uses artist introspection of docstrings to support properties.
All properties that you want to support through `~.pyplot.setp` and
`~.pyplot.getp` should have a ``set_property`` and ``get_property`` method in
the `~.matplotlib.artist.Artist` class. The setter methods use the docstring
with the ACCEPTS token to indicate the type of argument the method accepts.
e.g., in `.Line2D`:

.. code-block:: python

   # in lines.py
   def set_linestyle(self, linestyle):
       """
       Set the linestyle of the line

       ACCEPTS: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' | ' ' | '' ]
       """

Keyword arguments
-----------------
Since Matplotlib uses a lot of pass-through ``kwargs``, e.g., in every function
that creates a line (`~.pyplot.plot`, `~.pyplot.semilogx`, `~.pyplot.semilogy`,
etc...), it can be difficult for the new user to know which ``kwargs`` are
supported.  Matplotlib uses a docstring interpolation scheme to support
documentation of every function that takes a ``**kwargs``.  The requirements
are:

1. single point of configuration so changes to the properties don't
   require multiple docstring edits.

2. as automated as possible so that as properties change, the docs
   are updated automatically.

The function `matplotlib.artist.kwdoc` and the decorator
`matplotlib.docstring.dedent_interpd` facilitate this.  They combine Python
string interpolation in the docstring with the Matplotlib artist introspection
facility that underlies ``setp`` and ``getp``.  The ``kwdoc`` function gives
the list of properties as a docstring. In order to use this in another
docstring, first update the ``matplotlib.docstring.interpd`` object, as seen in
this example from `matplotlib.lines`:

.. code-block:: python

  # in lines.py
  docstring.interpd.update(Line2D=artist.kwdoc(Line2D))

Then in any function accepting `~.Line2D` pass-through ``kwargs``, e.g.,
`matplotlib.axes.Axes.plot`:

.. code-block:: python

  # in axes.py
  @docstring.dedent_interpd
  def plot(self, *args, **kwargs):
      """
      Some stuff omitted

      The kwargs are Line2D properties:
      %(Line2D)s

      kwargs scalex and scaley, if defined, are passed on
      to autoscale_view to determine whether the x and y axes are
      autoscaled; default True.  See Axes.autoscale_view for more
      information
      """

Note there is a problem for `~matplotlib.artist.Artist` ``__init__`` methods,
e.g., `matplotlib.patches.Patch.__init__`, which supports ``Patch`` ``kwargs``,
since the artist inspector cannot work until the class is fully defined and
we can't modify the ``Patch.__init__.__doc__`` docstring outside the class
definition.  There are some some manual hacks in this case, violating the
"single entry point" requirement above -- see the ``docstring.interpd.update``
calls in `matplotlib.patches`.

Adding figures
==============
Figures in the documentation are automatically generated from scripts.
It is not necessary to explicitly save the figure from the script; this will be
done automatically when the docs are built to ensure that the code that is
included runs and produces the advertised figure.

There are two options for where to put the code that generates a figure. If
you want to include a plot in the examples gallery, the code should be added to
the :file:`examples` directory. Plots from
the :file:`examples` directory can then be referenced through the symlink
``mpl_examples`` in the ``doc`` directory, e.g.:

.. code-block:: rst

  .. plot:: mpl_examples/lines_bars_and_markers/fill.py

Alternatively the plotting code can be placed directly in the docstring.
To include plots directly in docstrings, see the following guide:

Plot directive documentation
----------------------------

.. automodule:: matplotlib.sphinxext.plot_directive
   :no-undoc-members:

Examples
--------

The source of the files in the :file:`examples` directory are automatically run
and their output plots included in the documentation. To exclude an example
from having an plot generated insert "sgskip" somewhere in the filename.

Adding animations
=================

We have a Matplotlib Google/Gmail account with username ``mplgithub``
which we used to setup the github account but can be used for other
purposes, like hosting Google docs or Youtube videos.  You can embed a
Matplotlib animation in the docs by first saving the animation as a
movie using :meth:`matplotlib.animation.Animation.save`, and then
uploading to `matplotlib's Youtube
channel <https://www.youtube.com/user/matplotlib>`_ and inserting the
embedding string youtube provides like:

.. code-block:: rst

  .. raw:: html

     <iframe width="420" height="315"
       src="http://www.youtube.com/embed/32cjc6V0OZY"
       frameborder="0" allowfullscreen>
     </iframe>

An example save command to generate a movie looks like this

.. code-block:: python

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
        interval=25, blit=True, init_func=init)

    ani.save('double_pendulum.mp4', fps=15)

Contact Michael Droettboom for the login password to upload youtube videos of
google docs to the mplgithub account.

.. _referring-to-mpl-docs:

Referring to data files
=======================

In the documentation, you may want to include to a data file in the Matplotlib
sources, e.g., a license file or an image file from :file:`mpl-data`, refer to it via
a relative path from the document where the rst file resides, e.g.,
in :file:`users/navigation_toolbar.rst`, you can refer to the image icons with::

    .. image:: ../../lib/matplotlib/mpl-data/images/subplots.png

In the :file:`users` subdirectory, if you want to refer to a file in the
:file:`mpl-data` directory, you can use the symlink directory. For example,
from :file:`customizing.rst`::

    .. literalinclude:: ../../lib/matplotlib/mpl-data/matplotlibrc

One exception to this is when referring to the examples directory. Relative
paths are extremely confusing in the sphinx plot extensions, so it is easier
to simply include a symlink to the files at the top doc level directory.
This way, API documents like :meth:`matplotlib.pyplot.plot` can refer to the
examples in a known location.

In the top level :file:`doc` directory we have symlinks pointing to the
Matplotlib :file:`examples`:

.. code-block:: sh

    home:~/mpl/doc> ls -l mpl_*
    mpl_examples -> ../examples

So we can include plots from the examples dir using the symlink:

.. code-block:: rst

    .. plot:: mpl_examples/pylab_examples/simple_plot.py

.. _internal-section-refs:

Internal section references
===========================

To maximize internal consistency in section labeling and references,
use hyphen separated, descriptive labels for section references, e.g.:

.. code-block:: rst

    .. _howto-webapp:

and refer to it using  the standard reference syntax:

.. code-block:: rst

    See :ref:`howto-webapp`

Keep in mind that we may want to reorganize the contents later, so
please try to avoid top level names in references like ``user`` or ``devel``
or ``faq`` unless necessary, because for example the FAQ "what is a
backend?" could later become part of the users guide, so the label:

.. code-block:: rst

    .. _what-is-a-backend

is better than:

.. code-block:: rst

    .. _faq-backend

In addition, since underscores are widely used by Sphinx itself, please use
hyphens to separate words.

Section name formatting
=======================

For everything but top level chapters, please use ``Upper lower`` for
section titles, e.g., ``Possible hangups`` rather than ``Possible
Hangups``

Generating inheritance diagrams
===============================

Class inheritance diagrams can be generated with the
``inheritance-diagram`` directive.  To use it, provide the
directive with a number of class or module names (separated by
whitespace).  If a module name is provided, all classes in that module
will be used.  All of the ancestors of these classes will be included
in the inheritance diagram.

A single option is available: *parts* controls how many of parts in
the path to the class are shown.  For example, if *parts* == 1, the
class ``matplotlib.patches.Patch`` is shown as ``Patch``.  If *parts*
== 2, it is shown as ``patches.Patch``.  If *parts* == 0, the full
path is shown.

Example:

.. code-block:: rst

    .. inheritance-diagram:: matplotlib.patches matplotlib.lines matplotlib.text
       :parts: 2

.. inheritance-diagram:: matplotlib.patches matplotlib.lines matplotlib.text
   :parts: 2

.. _emacs-helpers:

Emacs helpers
=============

There is an emacs mode `rst.el
<http://docutils.sourceforge.net/tools/editors/emacs/rst.el>`_ which
automates many important ReST tasks like building and updating
table-of-contents, and promoting or demoting section headings.  Here
is the basic ``.emacs`` configuration:

.. code-block:: lisp

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

.. TODO: Add section about uploading docs

.. _Sphinx: http://www.sphinx-doc.org
.. _documentation: http://www.sphinx-doc.org/contents.html
.. _`inline markup`: http://www.sphinx-doc.org/markup/inline.html
.. _index: http://www.sphinx-doc.org/markup/para.html#index-generating-markup
.. _`Sphinx Gallery`: https://sphinx-gallery.readthedocs.io/en/latest/
