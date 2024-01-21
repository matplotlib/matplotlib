.. redirect-from:: /devel/documenting_mpl

.. _documenting-matplotlib:

===================
Write documentation
===================

Getting started
===============

General file structure
----------------------

All documentation is built from the :file:`doc/`.  The :file:`doc/`
directory contains configuration files for Sphinx and reStructuredText
(ReST_; ``.rst``) files that are rendered to documentation pages.

Documentation is created in three ways.  First, API documentation
(:file:`doc/api`) is created by Sphinx_ from
the docstrings of the classes in the Matplotlib library.  Except for
:file:`doc/api/api_changes/`,  ``.rst`` files in :file:`doc/api` are created
when the documentation is built.  See :ref:`writing-docstrings` below.

Second, our example pages, tutorials, and some of the narrative documentation
are created by `Sphinx Gallery`_.  Sphinx Gallery converts example Python files
to ``*.rst`` files with the result of Matplotlib plot calls as embedded images.
See :ref:`writing-examples-and-tutorials` below.

Third, Matplotlib has narrative docs written in ReST_ in subdirectories of
:file:`doc/users/`.  If you would like to add new documentation that is suited
to an ``.rst`` file rather than a gallery or tutorial example, choose an
appropriate subdirectory to put it in, and add the file to the table of
contents of :file:`index.rst` of the subdirectory.  See
:ref:`writing-rest-pages` below.

.. note::

  Don't directly edit the ``.rst`` files in :file:`doc/plot_types`,
  :file:`doc/gallery`,  :file:`doc/tutorials`, and :file:`doc/api`
  (excepting :file:`doc/api/api_changes/`).  Sphinx_ regenerates
  files in these directories when building documentation.

Set up the build
----------------

The documentation for Matplotlib is generated from reStructuredText (ReST_)
using the Sphinx_ documentation generation tool.

To build the documentation you will need to
:ref:`set up Matplotlib for development <installing_for_devs>`. Note in
particular the :ref:`additional dependencies <doc-dependencies>` required to
build the documentation.

Build the docs
--------------

The documentation sources are found in the :file:`doc/` directory.
The configuration file for Sphinx is :file:`doc/conf.py`. It controls which
directories Sphinx parses, how the docs are built, and how the extensions are
used. To build the documentation in html format, cd into :file:`doc/` and run:

.. code-block:: sh

   make html

Other useful invocations include

.. code-block:: sh

   # Build the html documentation, but skip generation of the gallery images to
   # save time.
   make html-noplot

   # Build the html documentation, but skip specific subdirectories.  If a gallery
   # directory is skipped, the gallery images are not generated.  The first
   # time this is run, it creates ``.mpl_skip_subdirs.yaml`` which can be edited
   # to add or remove subdirectories
   make html-skip-subdirs

   # Delete built files.  May help if you get errors about missing paths or
   # broken links.
   make clean

   # Build pdf docs.
   make latexpdf

The ``SPHINXOPTS`` variable is set to ``-W --keep-going`` by default to build
the complete docs but exit with exit status 1 if there are warnings.  To unset
it, use

.. code-block:: sh

   make SPHINXOPTS= html

You can use the ``O`` variable to set additional options:

* ``make O=-j4 html`` runs a parallel build with 4 processes.
* ``make O=-Dplot_formats=png:100 html`` saves figures in low resolution.

Multiple options can be combined, e.g. ``make O='-j4 -Dplot_formats=png:100'
html``.

On Windows, set the options as environment variables, e.g.:

.. code-block:: bat

   set SPHINXOPTS= & set O=-j4 -Dplot_formats=png:100 & make html

Show locally built docs
-----------------------

The built docs are available in the folder :file:`build/html`. A shortcut
for opening them in your default browser is:

.. code-block:: sh

   make show

.. _writing-rest-pages:

Write ReST pages
================

Most documentation is either in the docstrings of individual
classes and methods, in explicit ``.rst`` files, or in examples and tutorials.
All of these use the ReST_ syntax and are processed by Sphinx_.

The `Sphinx reStructuredText Primer
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ is
a good introduction into using ReST. More complete information is available in
the `reStructuredText reference documentation
<https://docutils.sourceforge.io/rst.html#reference-documentation>`_.

This section contains additional information and conventions how ReST is used
in the Matplotlib documentation.

Formatting and style conventions
--------------------------------

It is useful to strive for consistency in the Matplotlib documentation.  Here
are some formatting and style conventions that are used.

Section formatting
^^^^^^^^^^^^^^^^^^

Use `sentence case <https://apastyle.apa.org/style-grammar-guidelines/capitalization/sentence-case>`__
``Upper lower`` for section titles, e.g., ``Possible hangups`` rather than
``Possible Hangups``.

We aim to follow the recommendations from the
`Python documentation <https://devguide.python.org/documenting/#sections>`_
and the `Sphinx reStructuredText documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections>`_
for section markup characters, i.e.:

- ``#`` with overline, for parts. This is reserved for the main title in
  ``index.rst``. All other pages should start with "chapter" or lower.
- ``*`` with overline, for chapters
- ``=``, for sections
- ``-``, for subsections
- ``^``, for subsubsections
- ``"``, for paragraphs

This may not yet be applied consistently in existing docs.

Table formatting
^^^^^^^^^^^^^^^^
Given the size of the table and length of each entry, use:

+-------------+-------------------------------+--------------------+
|             | small table                   | large table        |
+-------------+-------------------------------+--------------------+
| short entry | `simple or grid table`_       | `grid table`_      |
+-------------+-------------------------------+--------------------+
| long entry  | `list table`_                 | `csv table`_       |
+-------------+-------------------------------+--------------------+

For more information, see `rst tables <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#tables>`_.

.. _`simple or grid table`: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#tables
.. _`grid table`: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#grid-tables
.. _`list table`: https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table
.. _`csv table`: https://docutils.sourceforge.io/docs/ref/rst/directives.html#csv-table-1

Function arguments
^^^^^^^^^^^^^^^^^^

Function arguments and keywords within docstrings should be referred to using
the ``*emphasis*`` role. This will keep Matplotlib's documentation consistent
with Python's documentation:

.. code-block:: rst

  Here is a description of *argument*

Do not use the ```default role```:

.. code-block:: rst

   Do not describe `argument` like this.  As per the next section,
   this syntax will (unsuccessfully) attempt to resolve the argument as a
   link to a class or method in the library.

nor the ````literal```` role:

.. code-block:: rst

   Do not describe ``argument`` like this.


.. _internal-section-refs:

Refer to other documents and sections
-------------------------------------

Sphinx_ supports internal references_:

==========  ===============  ===========================================
Role        Links target     Representation in rendered HTML
==========  ===============  ===========================================
|doc-dir|_  document         link to a page
|ref-dir|_  reference label  link to an anchor associated with a heading
==========  ===============  ===========================================

.. The following is a hack to have a link with literal formatting
   See https://stackoverflow.com/a/4836544

.. |doc-dir| replace:: ``:doc:``
.. _doc-dir: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-doc
.. |ref-dir| replace:: ``:ref:``
.. _ref-dir: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-ref

Examples:

.. code-block:: rst

   See the :doc:`/users/installing/index`

   See the tutorial :ref:`quick_start`

   See the example :doc:`/gallery/lines_bars_and_markers/simple_plot`

will render as:

  See the :doc:`/users/installing/index`

  See the tutorial :ref:`quick_start`

  See the example :doc:`/gallery/lines_bars_and_markers/simple_plot`

Sections can also be given reference labels.  For instance from the
:doc:`/users/installing/index` link:

.. code-block:: rst

   .. _clean-install:

   How to completely remove Matplotlib
   ===================================

   Occasionally, problems with Matplotlib can be solved with a clean...

and refer to it using the standard reference syntax:

.. code-block:: rst

   See :ref:`clean-install`

will give the following link: :ref:`clean-install`

To maximize internal consistency in section labeling and references,
use hyphen separated, descriptive labels for section references.
Keep in mind that contents may be reorganized later, so
avoid top level names in references like ``user`` or ``devel``
or ``faq`` unless necessary, because for example the FAQ "what is a
backend?" could later become part of the users guide, so the label:

.. code-block:: rst

   .. _what-is-a-backend:

is better than:

.. code-block:: rst

   .. _faq-backend:

In addition, since underscores are widely used by Sphinx itself, use
hyphens to separate words.

.. _referring-to-other-code:

Refer to other code
-------------------

To link to other methods, classes, or modules in Matplotlib you can use
back ticks, for example:

.. code-block:: rst

  `matplotlib.collections.LineCollection`

generates a link like this: `matplotlib.collections.LineCollection`.

*Note:* We use the sphinx setting ``default_role = 'obj'`` so that you don't
have to use qualifiers like ``:class:``, ``:func:``, ``:meth:`` and the likes.

Often, you don't want to show the full package and module name. As long as the
target is unambiguous you can simply leave them out:

.. code-block:: rst

  `.LineCollection`

and the link still works: `.LineCollection`.

If there are multiple code elements with the same name (e.g. ``plot()`` is a
method in multiple classes), you'll have to extend the definition:

.. code-block:: rst

  `.pyplot.plot` or `.Axes.plot`

These will show up as `.pyplot.plot` or `.Axes.plot`. To still show only the
last segment you can add a tilde as prefix:

.. code-block:: rst

  `~.pyplot.plot` or `~.Axes.plot`

will render as `~.pyplot.plot` or `~.Axes.plot`.

Other packages can also be linked via
`intersphinx <http://www.sphinx-doc.org/en/master/ext/intersphinx.html>`_:

.. code-block:: rst

  `numpy.mean`

will return this link: `numpy.mean`.  This works for Python, Numpy, Scipy,
and Pandas (full list is in :file:`doc/conf.py`).  If external linking fails,
you can check the full list of referenceable objects with the following
commands::

  python -m sphinx.ext.intersphinx 'https://docs.python.org/3/objects.inv'
  python -m sphinx.ext.intersphinx 'https://numpy.org/doc/stable/objects.inv'
  python -m sphinx.ext.intersphinx 'https://docs.scipy.org/doc/scipy/objects.inv'
  python -m sphinx.ext.intersphinx 'https://pandas.pydata.org/pandas-docs/stable/objects.inv'

.. _rst-figures-and-includes:

Include figures and files
-------------------------

Image files can directly included in pages with the ``image::`` directive.
e.g., :file:`tutorials/intermediate/constrainedlayout_guide.py` displays
a couple of static images::

  # .. image:: /_static/constrained_layout_1b.png
  #    :align: center


Files can be included verbatim.  For instance the ``LICENSE`` file is included
at :ref:`license-agreement` using ::

    .. literalinclude:: ../../LICENSE/LICENSE

The examples directory is copied to :file:`doc/gallery` by sphinx-gallery,
so plots from the examples directory can be included using

.. code-block:: rst

    .. plot:: gallery/lines_bars_and_markers/simple_plot.py

Note that the python script that generates the plot is referred to, rather than
any plot that is created.  Sphinx-gallery will provide the correct reference
when the documentation is built.

Tools for writing mathematical expressions
------------------------------------------

In most cases, you will likely want to use one of `Sphinx's builtin Math
extensions <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`__.
In rare cases we want the rendering of the mathematical text in the
documentation html to exactly match with the rendering of the mathematical
expression in the Matplotlib figure. In these cases, you can use the
`matplotlib.sphinxext.mathmpl` Sphinx extension (See also the
:doc:`../users/explain/text/mathtext` tutorial.)

.. _writing-docstrings:

Write docstrings
================

Most of the API documentation is written in docstrings. These are comment
blocks in source code that explain how the code works.

.. note::

   Some parts of the documentation do not yet conform to the current
   documentation style. If in doubt, follow the rules given here and not what
   you may see in the source code. Pull requests updating docstrings to
   the current style are very welcome.

All new or edited docstrings should conform to the `numpydoc docstring guide`_.
Much of the ReST_ syntax discussed above (:ref:`writing-rest-pages`) can be
used for links and references.  These docstrings eventually populate the
:file:`doc/api` directory and form the reference documentation for the
library.

Example docstring
-----------------

An example docstring looks like:

.. code-block:: python

    def hlines(self, y, xmin, xmax, colors=None, linestyles='solid',
               label='', **kwargs):
        """
        Plot horizontal lines at each *y* from *xmin* to *xmax*.

        Parameters
        ----------
        y : float or array-like
            y-indexes where to plot the lines.

        xmin, xmax : float or array-like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have the same length.

        colors : list of colors, default: :rc:`lines.color`

        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, optional

        label : str, default: ''

        Returns
        -------
        `~matplotlib.collections.LineCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs :  `~matplotlib.collections.LineCollection` properties.

        See Also
        --------
        vlines : vertical lines
        axhline : horizontal line across the Axes
        """

See the `~.Axes.hlines` documentation for how this renders.

The Sphinx_ website also contains plenty of documentation_ concerning ReST
markup and working with Sphinx in general.

Formatting conventions
----------------------

The basic docstring conventions are covered in the `numpydoc docstring guide`_
and the Sphinx_ documentation.  Some Matplotlib-specific formatting conventions
to keep in mind:

Quote positions
^^^^^^^^^^^^^^^

The quotes for single line docstrings are on the same line (pydocstyle D200)::

    def get_linewidth(self):
        """Return the line width in points."""

The quotes for multi-line docstrings are on separate lines (pydocstyle D213)::

        def set_linestyle(self, ls):
        """
        Set the linestyle of the line.

        [...]
        """

Function arguments
^^^^^^^^^^^^^^^^^^

Function arguments and keywords within docstrings should be referred to
using the ``*emphasis*`` role. This will keep Matplotlib's documentation
consistent with Python's documentation:

.. code-block:: rst

  If *linestyles* is *None*, the default is 'solid'.

Do not use the ```default role``` or the ````literal```` role:

.. code-block:: rst

  Neither `argument` nor ``argument`` should be used.


Quotes for strings
^^^^^^^^^^^^^^^^^^

Matplotlib does not have a convention whether to use single-quotes or
double-quotes.  There is a mixture of both in the current code.

Use simple single or double quotes when giving string values, e.g.

.. code-block:: rst

  If 'tight', try to figure out the tight bbox of the figure.

  No ``'extra'`` literal quotes.

The use of extra literal quotes around the text is discouraged. While they
slightly improve the rendered docs, they are cumbersome to type and difficult
to read in plain-text docs.

Parameter type descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main goal for parameter type descriptions is to be readable and
understandable by humans. If the possible types are too complex use a
simplification for the type description and explain the type more
precisely in the text.

Generally, the `numpydoc docstring guide`_ conventions apply. The following
rules expand on them where the numpydoc conventions are not specific.

Use ``float`` for a type that can be any number.

Use ``(float, float)`` to describe a 2D position. The parentheses should be
included to make the tuple-ness more obvious.

Use ``array-like`` for homogeneous numeric sequences, which could
typically be a numpy.array. Dimensionality may be specified using ``2D``,
``3D``, ``n-dimensional``. If you need to have variables denoting the
sizes of the dimensions, use capital letters in brackets
(``(M, N) array-like``). When referring to them in the text they are easier
read and no special formatting is needed. Use ``array`` instead of
``array-like`` for return types if the returned object is indeed a numpy array.

``float`` is the implicit default dtype for array-likes. For other dtypes
use ``array-like of int``.

Some possible uses::

  2D array-like
  (N,) array-like
  (M, N) array-like
  (M, N, 3) array-like
  array-like of int

Non-numeric homogeneous sequences are described as lists, e.g.::

  list of str
  list of `.Artist`

Reference types
^^^^^^^^^^^^^^^

Generally, the rules from referring-to-other-code_ apply. More specifically:

Use full references ```~matplotlib.colors.Normalize``` with an
abbreviation tilde in parameter types. While the full name helps the
reader of plain text docstrings, the HTML does not need to show the full
name as it links to it. Hence, the ``~``-shortening keeps it more readable.

Use abbreviated links ```.Normalize``` in the text.

.. code-block:: rst

   norm : `~matplotlib.colors.Normalize`, optional
        A `.Normalize` instance is used to scale luminance data to 0, 1.

Default values
^^^^^^^^^^^^^^

As opposed to the numpydoc guide, parameters need not be marked as
*optional* if they have a simple default:

- use ``{name} : {type}, default: {val}`` when possible.
- use ``{name} : {type}, optional`` and describe the default in the text if
  it cannot be explained sufficiently in the recommended manner.

The default value should provide semantic information targeted at a human
reader. In simple cases, it restates the value in the function signature.
If applicable, units should be added.

.. code-block:: rst

   Prefer:
       interval : int, default: 1000ms
   over:
       interval : int, default: 1000

If *None* is only used as a sentinel value for "parameter not specified", do
not document it as the default. Depending on the context, give the actual
default, or mark the parameter as optional if not specifying has no particular
effect.

.. code-block:: rst

   Prefer:
       dpi : float, default: :rc:`figure.dpi`
   over:
       dpi : float, default: None

   Prefer:
       textprops : dict, optional
           Dictionary of keyword parameters to be passed to the
           `~matplotlib.text.Text` instance contained inside TextArea.
   over:
       textprops : dict, default: None
           Dictionary of keyword parameters to be passed to the
           `~matplotlib.text.Text` instance contained inside TextArea.


``See also`` sections
^^^^^^^^^^^^^^^^^^^^^

Sphinx automatically links code elements in the definition blocks of ``See
also`` sections. No need to use backticks there::

   See Also
   --------
   vlines : vertical lines
   axhline : horizontal line across the Axes

Wrap parameter lists
^^^^^^^^^^^^^^^^^^^^

Long parameter lists should be wrapped using a ``\`` for continuation and
starting on the new line without any indent (no indent because pydoc will
parse the docstring and strip the line continuation so that indent would
result in a lot of whitespace within the line):

.. code-block:: python

  def add_axes(self, *args, **kwargs):
      """
      ...

      Parameters
      ----------
      projection : {'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', \
  'rectilinear'}, optional
          The projection type of the axes.

      ...
      """

Alternatively, you can describe the valid parameter values in a dedicated
section of the docstring.

rcParams
^^^^^^^^

rcParams can be referenced with the custom ``:rc:`` role:
:literal:`:rc:\`foo\`` yields ``rcParams["foo"] = 'default'``, which is a link
to the :file:`matplotlibrc` file description.

Setters and getters
-------------------

Artist properties are implemented using setter and getter methods (because
Matplotlib predates the Python `property` decorator).
By convention, these setters and getters are named ``set_PROPERTYNAME`` and
``get_PROPERTYNAME``; the list of properties thusly defined on an artist and
their values can be listed by the `~.pyplot.setp` and `~.pyplot.getp` functions.

The Parameters block of property setter methods is parsed to document the
accepted values, e.g. the docstring of `.Line2D.set_linestyle` starts with

.. code-block:: python

   def set_linestyle(self, ls):
       """
       Set the linestyle of the line.

       Parameters
       ----------
       ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
           etc.
       """

which results in the following line in the output of ``plt.setp(line)`` or
``plt.setp(line, "linestyle")``::

    linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

In some rare cases (mostly, setters which accept both a single tuple and an
unpacked tuple), the accepted values cannot be documented in such a fashion;
in that case, they can be documented as an ``.. ACCEPTS:`` block, e.g. for
`.axes.Axes.set_xlim`:

.. code-block:: python

   def set_xlim(self, left=None, right=None):
       """
       Set the x-axis view limits.

       Parameters
       ----------
       left : float, optional
           The left xlim in data coordinates. Passing *None* leaves the
           limit unchanged.

           The left and right xlims may also be passed as the tuple
           (*left*, *right*) as the first positional argument (or as
           the *left* keyword argument).

           .. ACCEPTS: (bottom: float, top: float)

       right : float, optional
           etc.
       """

Note that the leading ``..`` makes the ``.. ACCEPTS:`` block a reST comment,
hiding it from the rendered docs.

Keyword arguments
-----------------

.. note::

  The information in this section is being actively discussed by the
  development team, so use the docstring interpolation only if necessary.
  This section has been left in place for now because this interpolation
  is part of the existing documentation.

Since Matplotlib uses a lot of pass-through ``kwargs``, e.g., in every function
that creates a line (`~.pyplot.plot`, `~.pyplot.semilogx`, `~.pyplot.semilogy`,
etc.), it can be difficult for the new user to know which ``kwargs`` are
supported.  Matplotlib uses a docstring interpolation scheme to support
documentation of every function that takes a ``**kwargs``.  The requirements
are:

1. single point of configuration so changes to the properties don't
   require multiple docstring edits.

2. as automated as possible so that as properties change, the docs
   are updated automatically.

The ``@_docstring.interpd`` decorator implements this.  Any function accepting
`.Line2D` pass-through ``kwargs``, e.g., `matplotlib.axes.Axes.plot`, can list
a summary of the `.Line2D` properties, as follows:

.. code-block:: python

  # in axes.py
  @_docstring.interpd
  def plot(self, *args, **kwargs):
      """
      Some stuff omitted

      Other Parameters
      ----------------
      scalex, scaley : bool, default: True
          These parameters determine if the view limits are adapted to the
          data limits. The values are passed on to `autoscale_view`.

      **kwargs : `.Line2D` properties, optional
          *kwargs* are used to specify properties like a line label (for
          auto legends), linewidth, antialiasing, marker face color.
          Example::

          >>> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)
          >>> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')

          If you specify multiple lines with one plot call, the kwargs apply
          to all those lines. In case the label object is iterable, each
          element is used as labels for each set of data.

          Here is a list of available `.Line2D` properties:

          %(Line2D:kwdoc)s
      """

The ``%(Line2D:kwdoc)`` syntax makes ``interpd`` lookup an `.Artist` subclass
named ``Line2D``, and call `.artist.kwdoc` on that class.  `.artist.kwdoc`
introspects the subclass and summarizes its properties as a substring, which
gets interpolated into the docstring.

Note that this scheme does not work for decorating an Artist's ``__init__``, as
the subclass and its properties are not defined yet at that point.  Instead,
``@_docstring.interpd`` can be used to decorate the class itself -- at that
point, `.kwdoc` can list the properties and interpolate them into
``__init__.__doc__``.


Inherit docstrings
------------------

If a subclass overrides a method but does not change the semantics, we can
reuse the parent docstring for the method of the child class. Python does this
automatically, if the subclass method does not have a docstring.

Use a plain comment ``# docstring inherited`` to denote the intention to reuse
the parent docstring. That way we do not accidentally create a docstring in
the future::

    class A:
        def foo():
            """The parent docstring."""
            pass

    class B(A):
        def foo():
            # docstring inherited
            pass


.. _docstring-adding-figures:

Add figures
-----------

As above (see :ref:`rst-figures-and-includes`), figures in the examples gallery
can be referenced with a ``.. plot::`` directive pointing to the python script
that created the figure.  For instance the `~.Axes.legend` docstring references
the file :file:`examples/text_labels_and_annotations/legend.py`:

.. code-block:: python

    """
    ...

    Examples
    --------

    .. plot:: gallery/text_labels_and_annotations/legend.py
    """

Note that ``examples/text_labels_and_annotations/legend.py`` has been mapped to
``gallery/text_labels_and_annotations/legend.py``, a redirection that may be
fixed in future re-organization of the docs.

Plots can also be directly placed inside docstrings.  Details are in
:doc:`/api/sphinxext_plot_directive_api`.  A short example is:

.. code-block:: python

    """
    ...

    Examples
    --------

    .. plot::
       import matplotlib.image as mpimg
       img = mpimg.imread('_static/stinkbug.png')
       imgplot = plt.imshow(img)
    """

An advantage of this style over referencing an example script is that the
code will also appear in interactive docstrings.

.. _writing-examples-and-tutorials:

Write examples and tutorials
============================

Examples and tutorials are Python scripts that are run by `Sphinx Gallery`_.
Sphinx Gallery finds ``*.py`` files in source directories and runs the files to
create images and narrative that are embedded in ``*.rst`` files in a build
location of the :file:`doc/` directory.  Files in the build location should not
be directly edited as they will be overwritten by Sphinx gallery. Currently
Matplotlib has four galleries as follows:

===============================  ==========================
Source location                  Build location
===============================  ==========================
:file:`galleries/plot_types`     :file:`doc/plot_types`
:file:`galleries/examples`       :file:`doc/gallery`
:file:`galleries/tutorials`      :file:`doc/tutorials`
:file:`galleries/users_explain`  :file:`doc/users/explain`
===============================  ==========================

The first three are traditional galleries.  The last,
:file:`galleries/users_explain`, is a mixed gallery where some of the files are
raw ``*.rst`` files and some are ``*.py`` files; Sphinx Gallery just copies
these ``*.rst`` files from the source location to the build location (see
:ref:`raw_restructured_gallery`, below).

In the Python files, to exclude an example from having a plot generated, insert
"sgskip" somewhere in the filename.

Format examples
---------------

The format of these files is relatively straightforward.  Properly
formatted comment blocks are treated as ReST_ text, the code is
displayed, and figures are put into the built page.  Matplotlib uses the
``# %%`` section separator so that IDEs will identify "code cells" to make
it easy to re-run sub-sections of the example.

For instance the example :doc:`/gallery/lines_bars_and_markers/simple_plot`
example is generated from
:file:`/galleries/examples/lines_bars_and_markers/simple_plot.py`, which looks
like:

.. code-block:: python

    """
    ===========
    Simple Plot
    ===========

    Create a simple plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()
    plt.show()

The first comment block is treated as ReST_ text.  The other comment blocks
render as comments in :doc:`/gallery/lines_bars_and_markers/simple_plot`.

Tutorials are made with the exact same mechanism, except they are longer, and
typically have more than one comment block (i.e. :ref:`quick_start`).  The
first comment block can be the same as the example above.  Subsequent blocks of
ReST text are delimited by the line ``# %%`` :

.. code-block:: python

    """
    ===========
    Simple Plot
    ===========

    Create a simple plot.
    """
    ...
    ax.grid()
    plt.show()

    # %%
    # Second plot
    # ===========
    #
    # This is a second plot that is very nice

    fig, ax = plt.subplots()
    ax.plot(np.sin(range(50)))

In this way text, code, and figures are output in a "notebook" style.

.. _sample-data:

Sample data
-----------

When sample data comes from a public dataset, please cite the source of the
data. Sample data should be written out in the code. When this is not
feasible, the data can be loaded using `.cbook.get_sample_data`.

.. code-block:: python

    import matplotlib.cbook as cbook
    fh = cbook.get_sample_data('mydata.dat')


If the data is too large to be included in the code, it should be added to
:file:`lib/matplotlib/mpl-data/sample_data/`

Create mini-gallery
-------------------

The showcased Matplotlib functions should be listed in an admonition at the
bottom as follows

.. code-block:: python

    # %%
    #
    # .. admonition:: References
    #
    #    The use of the following functions, methods, classes and modules is shown
    #    in this example:
    #
    #    - `matplotlib.axes.Axes.fill` / `matplotlib.pyplot.fill`
    #    - `matplotlib.axes.Axes.axis` / `matplotlib.pyplot.axis`

This allows sphinx-gallery to place an entry to the example in the
mini-gallery of the mentioned functions. Whether or not a function is mentioned
here should be decided depending on if a mini-gallery link prominently helps
to illustrate that function; e.g. mention ``matplotlib.pyplot.subplots`` only
in examples that are about laying out subplots, not in every example that uses
it.

Functions that exist in ``pyplot`` as well as in Axes or Figure should mention
both references no matter which one is used in the example code. The ``pyplot``
reference should always be the second to mention; see the example above.


Order examples
--------------

The order of the sections of the :ref:`tutorials` and the :ref:`gallery`, as
well as the order of the examples within each section are determined in a
two step process from within the :file:`/doc/sphinxext/gallery_order.py`:

* *Explicit order*: This file contains a list of folders for the section order
  and a list of examples for the subsection order. The order of the items
  shown in the doc pages is the order those items appear in those lists.
* *Implicit order*: If a folder or example is not in those lists, it will be
  appended after the explicitly ordered items and all of those additional
  items will be ordered by pathname (for the sections) or by filename
  (for the subsections).

As a consequence, if you want to let your example appear in a certain
position in the gallery, extend those lists with your example.
In case no explicit order is desired or necessary, still make sure
to name your example consistently, i.e. use the main function or subject
of the example as first word in the filename; e.g. an image example
should ideally be named similar to :file:`imshow_mynewexample.py`.

.. _raw_restructured_gallery:

Raw restructured text files in the gallery
------------------------------------------

`Sphinx Gallery`_ folders usually consist of a ``README.txt`` and a series of
Python source files that are then translated to an ``index.rst`` file and a
series of ``example_name.rst`` files in the :file:`doc/` subdirectories.
However, Sphinx Gallery also allows raw ``*.rst`` files to be passed through a
gallery (see `Manually passing files`_ in the Sphinx Gallery documentation). We
use this feature in :file:`galleries/users_explain`, where, for instance,
:file:`galleries/users_explain/colors` is a regular Sphinx Gallery
subdirectory, but  :file:`galleries/users_explain/artists` has a mix of
``*.rst`` and ``*py`` files.  For mixed subdirectories like this, we must add
any ``*.rst`` files to a ``:toctree:``, either in the ``README.txt`` or in a
manual ``index.rst``.

Miscellaneous
=============

Move documentation
------------------

Sometimes it is desirable to move or consolidate documentation.  With no
action this will lead to links either going dead (404) or pointing to old
versions of the documentation.  Preferable is to replace the old page
with an html refresh that immediately redirects the viewer to the new
page. So, for example we move ``/doc/topic/old_info.rst`` to
``/doc/topic/new_info.rst``.  We remove ``/doc/topic/old_info.rst`` and
in ``/doc/topic/new_info.rst`` we insert a ``redirect-from`` directive that
tells sphinx to still make the old file with the html refresh/redirect in it
(probably near the top of the file to make it noticeable)

.. code-block:: rst

   .. redirect-from:: /topic/old_info

In the built docs this will yield an html file
``/build/html/topic/old_info.html`` that has a refresh to ``new_info.html``.
If the two files are in different subdirectories:

.. code-block:: rst

   .. redirect-from:: /old_topic/old_info2

will yield an html file ``/build/html/old_topic/old_info2.html`` that has a
(relative) refresh to ``../topic/new_info.html``.

Use the full path for this directive, relative to the doc root at
``https://matplotlib.org/stable/``.  So ``/old_topic/old_info2`` would be
found by users at ``http://matplotlib.org/stable/old_topic/old_info2``.
For clarity, do not use relative links.


.. _inheritance-diagrams:

Generate inheritance diagrams
-----------------------------

Class inheritance diagrams can be generated with the Sphinx
`inheritance-diagram`_ directive.

.. _inheritance-diagram: https://www.sphinx-doc.org/en/master/usage/extensions/inheritance.html

Example:

.. code-block:: rst

    .. inheritance-diagram:: matplotlib.patches matplotlib.lines matplotlib.text
       :parts: 2

.. inheritance-diagram:: matplotlib.patches matplotlib.lines matplotlib.text
   :parts: 2


Navbar and style
----------------

Matplotlib has a few subprojects that share the same navbar and style, so these
are centralized as a sphinx theme at
`mpl_sphinx_theme <https://github.com/matplotlib/mpl-sphinx-theme>`_.  Changes to the
style or topbar should be made there to propagate across all subprojects.

.. TODO: Add section about uploading docs

.. _ReST: https://docutils.sourceforge.io/rst.html
.. _Sphinx: http://www.sphinx-doc.org
.. _documentation: https://www.sphinx-doc.org/en/master/contents.html
.. _index: http://www.sphinx-doc.org/markup/para.html#index-generating-markup
.. _`Sphinx Gallery`: https://sphinx-gallery.readthedocs.io/en/latest/
.. _references: https://www.sphinx-doc.org/en/stable/usage/restructuredtext/roles.html
.. _`numpydoc docstring guide`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _`Manually passing files`: https://sphinx-gallery.github.io/stable/configuration.html#manually-passing-files
