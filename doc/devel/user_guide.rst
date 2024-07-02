.. _user-guide-content-detail:

User guide content guide
========================

.. summary-begin

The user guide explains what the reader needs to know to *make* and *customize*
visualizations, and *develop* tools and downstream libraries:

+-----------+---------------+-----------------------------------------------------+
|  section  |     focus     | purpose                                             |
+===========+===============+=====================================================+
| Make      | Features      | Provide overview of library features                |
+-----------+---------------+-----------------------------------------------------+
| Customize | Components    | Review what each component does                     |
+-----------+---------------+-----------------------------------------------------+
| Develop   | Architecture  | Explain implementation details and design decisions |
+-----------+---------------+-----------------------------------------------------+

Each section builds on the previous to expand the reader's conceptal understanding of
the library so that the user can develop a mental model (schema) of how Matplotlib
operates. For example, *make* introduces the concept of making figures,
*customize* explains what the user can do with figures, and *develop* discusses
how the Figure object works in the context of the other parts of the library. The
goal is that understanding how the features, components, and architecture fit
together empowers the user to piece together solutions from existing
documentation.

.. summary-end

.. _content-user-guide-make:

Make
-----

The goal of this section is to provide an overview of using Matplotlib to make things.
Therefore, content should be generalized rather than focused on specific tasks, e.g.
*what is the usage pattern for making plots?* rather than *How do I make a squiggly
yellow line?*


Audience
^^^^^^^^

This section of the user guide assumes that it is the readers first introduction to
Matplotlib. Therefore, the reader may not yet know what Matplotlib calls a given v
isualization task nor how any task is accomplished in Matplotlib. Those documents should
first introduce or define the object/module/concept that it is discussing and why it is
important for the reader. e.g.:

  A matplotlib visualization is managed by a `~.Figure` object onto which is attached
  one or more `~.matplotlib.axes.Axes` objects. Each Axes object manages a plotting
  region. The coordinate system of this region is managed by `~.matplotlib.axis.Axis`
  objects; often these are x Axis and y Axis objects.


While a lot more can be said for each of the objects introduced here, that level of
detail is out of scope for the getting started guide. Here the goal is to provide just
enough vocabulary so that readers can follow along with the rest of the guide.

Generally this section of the user guide should be written with the assumption that it
is read linearlly.

Format
^^^^^^

When possible, the material is introduced in tightly scoped sections that build on top
of each other, using teaching strategies called `chunking and scaffolding`_.
Chunking is aimed at reducing `cognitive load`_ by keeping the focus of each section
relatively small, and scaffolding lays out instructional material such that each section
builds on the previous one.

.. code-block:: rst
  :caption: Example: make

  A matplotlib visualization is managed by a `~.Figure` object onto which is attached
  one or more `~.matplotlib.axes.Axes` objects. Each Axes object manages a plotting
  region. The coordinate system of this region is managed by `~.matplotlib.axis.Axis`
  objects; often these are x Axis and y Axis objects.

  Generally, visualizations are created by first creating a `~.Figure` and
  `~.matplotlib.axes.Axes`, and then calling plotting methods on that axes:

  For example, to make a line plot::

    fig, ax = plt.subplots()
    ax.plot([1,2,3], [2,1,2])

  Making a bar plot follows the same format::

    fig, ax = plt.subplots()
    ax.bar([1,2,3], [2,1,2])

  As does an image, with the input changed to a 2D array::

    fig, ax = plt.subplots()
    ax.imshow([[1,2,3], [2,1,2]])

Here this example is very tightly scoped to how to make a single plot. The goal of these
small and repetitive examples is to reinforce that making visualizations, regardless of
chart type, follows the same basic pattern of: 1) create figure and axes, 2) call
plotting method with data. For some topics and examples, it may be preferable to take
the inverse approach: start with a complex example and then unpack it into its pieces.

In general, this approach is aimed at helping the reader develop a model of the concept
by first defining it and then illustrating how that understanding facilitates better use
of the library. In keeping with this goal, documents should also be tightly focused on
one topic and link out to related material, e.g. *insert explanation*

.. _`chunking and scaffolding`: https://www.tacoma.uw.edu/digital-learning/chunking-scaffolding-pacing
.. _`cognitive load`: https://carpentries.github.io/instructor-training/05-memory.html


.. _content-user-guide-customize:

Customize
---------

The goal of this section is to explain how the components of the library can be used to
customize visualizations.

Audience
^^^^^^^^

This section assumes that the reader understands how to make visualizations using
Matplotlib but does not yet have a solid mental model of what the various components of
the library are responsible for or how they fit together. For example, they know how to
set the color of a heatmap using the ``cmap``` keyword and are now interested in
learning about colormaps and normalization. The documents in this section are generally
assumed to be accessed independent of each other unless cross referenced.

Format
^^^^^^

Similar to the :ref:`content-user-guide-make` section, the documents in this section
aim to provide a review of using the component and generally aim to do so in a
scaffolded and chunked manner.

.. code-block:: rst
  :caption: Example: customize

  A `~.Figure` is roughly the total drawing area and keeps track of the child
  `~matplotlib.axes.Axes`, figure related artists such as titles, figure legends,
  colorbars, and nested subfigures.

  Generally Figures are created through helper methods that also create Axes objects,
  as discussed in :ref:`arranging_axes`. Here we create a Figure without an Axes to
  isolate manipulating Figure object::

    fig = plt.figure()

  The Figure size on the screen is set by figsize and dpi; figsize is the
  (width, height) of the Figure in inches or units of 72 typographic points, while
  dpi is how many pixels per inch the figure will be rendered at. Here we set the
  figure size to be a :math:`5 \times 5` square ::

    fig = plt.figure(figsize=(5,5))

  To make your Figures appear on the screen at the physical size you requested, you
  should set dpi to the same dpi as your graphics system.

Here this example unpacks the `~.Figure` object that was briefly mentioned in the
previous example. It describes the `~.Figure` object in more detail, then explains how
to create an object and links out to the guide that discusses other ways of doing so,
then shows the frequent use case of changing the figure size. The goal of this example
is to help the reader understand what they can do with a `.Figure` object.

.. _content-user-guide-extend:

Develop
-------

The goal of this section is to explain the design considerations and implementation
details necessary for building downstream libraries using Matplotlib.

Audience
^^^^^^^^
This section assumes that the reader is a downstream library developer. This reader has
a solid mental model of the library and needs to understand underlying design decisions
and implementation details so that they can build extensions such as custom Artists or
projections. The documents in this section are generally assumed to be accessed
independent of each other unless cross referenced.

Format
^^^^^^
Like the other sections, explanations into buildable well scoped chunks. As mentioned,
the primary difference is that the content will focus on implementation details because
the goal is to explain how the parts of the library work.

.. code-block:: rst
  :caption: Example: develop

  ``pyplot.figure()`` can create a new `~.Figure` instance and associate it with an
  instance of a backend-provided canvas class, itself hosted in an instance of a
  backend-provided manager class.

Building on the assumption that the user is familiar with the Figure object, this
section dives straight into what the ``pyplot.figure()`` method does when creating
Figures.
