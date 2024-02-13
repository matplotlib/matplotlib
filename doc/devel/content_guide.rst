.. _documenting-content:

***************************
Documentation content guide
***************************

These guidelines aim to improve the consistency, cohesiveness, and organization of the
:ref: documentation concerning using the library by broadly articulating the intended purpose,
scope, and audience of each of the following sections:

:ref:`plot_types`
    | Summary of chart types that are implemented as high-level API

:ref:`User guide <users-guide-index>`
    | Explanations of features, components, and architecture.

:ref:`tutorials`
    | Lessons on developing visualizations

:ref:`examples-index`
    | Demonstrations of specific library features

:ref:`api-reference`
    | Descriptions of the public modules, objects, methods, and functions.


.. note::

    While parts of the current documentation may not yet align with these guidelines,
    we expect that new documentation contributions will also bring existing documentation
    into alignment.

    .. note: based on note in  https://matplotlib.org/3.7.3/devel/coding_guide.html



Summary
==========

This content guide is heavily influenced by the `Diátaxis <https://diataxis.fr/>`_
framework for technical documentation; this framework proposes that all
technical documentation is roughly one of 4 types - tutorials, how-to-guides,
technical reference, and explanation - and that the appropriate type is
determined by whether the document focuses on *what* is x? (cognition) or *how*
to do x?(action), and whether the document's purpose is *acquiring* (learning)
or *applying* (using) skills.

Broadly, our documentation as viewed through a diátaxis lens:

.. list-table::
   :header-rows: 1
   :widths: 20 30 20 30

   * - section
     - goal
     - type
     - example
   * - :ref:`plot_types`
     - View available chart types.
     - | `Reference <https://diataxis.fr/reference/>`_
       | what, use
     - :ref:`sphx_glr_plot_types_stats_pie.py`
   * - :ref:`User guide <users-guide-index>`
     -  Understand features, components, and architecture
     - | `Explanation <https://diataxis.fr/explanation/>`_
       | what, learn
     - :ref:`annotations`
   * - :ref:`tutorials`
     - Work through the stages of building a visualization.
     - | `Tutorials <https://diataxis.fr/tutorials/>`_
       | how, learn
     - :ref:`sphx_glr_gallery_pie_and_polar_charts_pie_and_donut_labels.py`
   * - :ref:`examples-index`
     -  Execute a visualization task.
     - | `How-to guides <https://diataxis.fr/how-to-guides/>`_
       | how, use
     - :ref:`sphx_glr_gallery_text_labels_and_annotations_rainbow_text.py`
   * - :ref:`api-reference`
     -  Look up the input/output/behavior of public API.
     - | `Reference <https://diataxis.fr/reference/>`_
       | what, use
     - `.Axes.annotate`

Audience
--------
The Matplotlib audience encompasses a wide spectrum of readers, from users who are first
getting introduced to using Matplotlib through the documentation to experienced developers
who want to make something extremely customized. Instead of trying to write for the
entire spectrum, each document should identify its band so that reader can assess
whether the document is appropriate for them. The documentation should generally use the
leveling of :doc:`tags <tag_guidelines>` and :ref:`issues <new_contributors>`, meaning
that the audienced is identified based on how much contextual understanding of
Matplotlib is a pre-requisite for following along with the document.

Documents can communicate this leveling through tags, location in the docs, and in text
as appropriate. For example:

  This guide assumes that the reader understands the general concept of Artists, as
  explained in :ref:`users_artists`


Scope
-----
Many concepts in Matplotlib assume a grounding in visualization, statistics, Python
programming, and other topics to understand how they work. These concepts should be
contextualized using common terminology, but the focus for all documents should not
stray from the Matplotlib topic. For example:

  Some of the path components require multiple vertices to specify them: for example
  CURVE 3 is a `Bézier curve <https://en.wikipedia.org/wiki/B%C3%A9zier_curve>`_ with
  one control point and one end point, and CURVE4 has three vertices for the two
  control points and the end point. The example below shows a CURVE4 Bézier spline --
  the Bézier curve will be contained in the convex hull of the start point, the two
  control points, and the end point


This explanation of drawing a curve using control points from :ref:`paths` never
explicitly defines a *Bézier curve*, instead linking out to an external
reference. This explanation is written in a manner where the definition of
*Bézier curve* can be inferred from context and also understanding is not harmed if the
reader does not infer the definition.


.. _content-plot-types:

Plot types gallery
==================

The plot type gallery displays a selection of the common visualization techniques that
are implemented in Matplotlib. The gallery is heavily curated and tightly scoped to the
plotting methods on `matplotlib.axes.Axes` so additions are generally discouraged.

Format
------
* very short: 5-10 lines
* self explanatory data
* little to no customization.

.. _content-user-guide:

User guide
==========

.. toctree::
   :hidden:

   user_guide

.. include:: user_guide.rst
   :start-after: summary-begin
   :end-before: summary-end

For more details, see :ref:`user-guide-content-detail`

.. _content-tutorials:

Tutorials
=========

The goal of the tutorials is to guide the reader through the stages of using
Matplotlib to build a specific visualization. A tutorial describes what is
happening at each stage of executing a visualization task and links to the
user guide and API for explanations.

Format
------

The sample tutorial here is trivial to highlight the step-wise format:

#. First we start by stating the objective:

   .. code-block:: rst

      The goal of this tutorial is to animate a sin wave.

#. Then we describe what needs to be in place to do the task:

   .. code-block:: rst

      First lets generate a sin wave::

        x = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(x)

      Then we plot an empty line and capture the returned line object::

        fig, ax = plt.subplot()
        l, _ = ax.plot(0,0)

#. Next we walk the reader through each step of executing the task:

   .. code-block:: rst

      Next we write a function that changes the plot on each frame. Here we grow
      the sin curve on each update by setting the new x and y data::

        def update(frame):

          l.set_data(x[:i], y[:i])

      Lastly we add an animation writer. Here we specify 30 milliseconds between each
      of the 40 frames::

        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

#. Then we summarize by putting all the steps together:

   .. code-block:: rst

      Now lets put it all together so we can plot an animated sin curve::

        x = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(x)

        fig, ax = plt.subplot()
        l, _ = ax.plot(0,0)

        def update(frame):
            l.set_data(x[:i], y[:i])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

#. Finally, we close with a call to action to learn about the underlying concepts:

   .. code-block:: rst

      For more information on animations and lines, see:
      * :ref:`animations`
      * ``:ref:Line2d``

Please note that explanations of domain should be limited to providing
contextually necessary information, and tutorials that are heavily domain
specific may be more appropriate for the Scientific Python
`blog <https://blog.scientific-python.org/>`_.



.. _content-examples:

Examples gallery
================

The gallery of examples contains visual demonstrations of Matplotlib features. Gallery
examples exist so that readers can scan through visual examples. Unlike tutorials or
user guides, gallery examples teach by demonstration, rather than by instruction or
explanation.

Gallery examples should avoid instruction or excessive explanation except for brief
clarifying code comments. Instead, they can tag related concepts and/or link to relevant
tutorials or user guides.

Format
------

All :ref:`examples-index` should aim to follow the following format:

* Title: 1-6 words, descriptive of content
* Subtitle: 10-50 words, action-oriented description of the example subject
* Image: a clear demonstration of the subject, showing edge cases and different
  applications if possible
* Code + Text (optional): code, commented as appropriate + written text to add context
  if necessary

Example:

The ``bbox_intersect`` gallery example demonstrates the point of visual examples:

* this example is "messy" in that it's hard to categorize, but the gallery is the right
  spot for it because it makes sense to find it by visual search
* :ref:`sphx_glr_gallery_misc_bbox_intersect.py`


.. _content-api:

API reference
=============

The API reference documentation describes the library interfaces, e.g. inputs, outputs,
and expected behavior. See :ref:`writing-docstrings` for guidance on writing docstrings.

The pages in :file:`doc/api` are purely technical definitions of layout; therefore new
API reference documentation should be added to the module docstrings. This placement
keeps all API reference documentation about a module in the same file.
