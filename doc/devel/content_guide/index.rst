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



Guidelines
==========

This content guide is heavily influenced by the `Diátaxis <https://diataxis.fr/>`_
framework for technical documentation; this framework proposes that all technical
documentation is roughly one of 4 types - tutorials, how-to-guides,
technical reference, and explanation - and that the appropriate type is determined
by whether *what* is x? (cognition) vs *how* to do x?(action) is being documented, and
whether the document's purpose is *acquiring* (learning) or *applying* (using) skills.

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

Detailed guidelines for each section are documented at their respective pages:

.. toctree::
  :maxdepth: 1

  plot_types
  user_guide
  tutorials
  examples
  api


Audience
========

The Matplotlib audience encompasses a wide spectrum of readers, from users who are first
getting introduced to using Matplotlib through the documentation to experienced developers
who want to make something extremely customized. Instead of trying to write for the
entire spectrum, each document should identify its band so that reader can assess
whether the document is appropriate for them. The documentation should generally use the
leveling of :doc:`tags <../tag_guidelines>` and :ref:`issues <new_contributors>`, meaning
that the audienced is identified based on how much contextual understanding of
Matplotlib is a pre-requisite for following along with the document.

Documents can communicate this leveling through tags, location in the docs, and in text
as appropriate. For example:

  This guide assumes that the reader understands the general concept of Artists, as
  explained in :ref:`users_artists`


Scope
=====
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
