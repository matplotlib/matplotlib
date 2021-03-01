
=========================
Documentation Style Guide
=========================

This guide contains best practices for the language and formatting of Matplotlib
documentation.

.. seealso::

    For more information about contributing, see the :ref:`documenting-matplotlib`
    section.

Expository language
===================

For explanatory writing, the following guidelines are for clear and concise
language use.

Terminology
-----------

There are several key terms in Matplotlib that are standards for 
reliability and consistency in documentation. They are not interchangeable.

+------------------+--------------------------+--------------+--------------+
| Term             | Description              | Correct      | Incorrect    |
+==================+==========================+==============+==============+
| Figure_          | Matplotlib working space | - *For       | - "The figure|
|                  | for programming.         |   Matplotlib |   is the     |
|                  |                          |   objects*:  |   working    |
|                  |                          |   Figure,    |   space for  |
|                  |                          |   "The Figure|   visuals."  |
|                  |                          |   is the     | - "Methods in|
|                  |                          |   working    |   the Figure |
|                  |                          |   space for  |   provide the|
|                  |                          |   the visual.|   visuals."  |
|                  |                          | - *Referring | - "The       |
|                  |                          |   to class*: |   Figure_ on |
|                  |                          |   Figure_ ,  |   the page is|
|                  |                          |   "Methods   |   static."   |
|                  |                          |   within the |              |
|                  |                          |   Figure_    |              |
|                  |                          |   provide the|              |
|                  |                          |   visuals."  |              |
|                  |                          | - *General   |              |
|                  |                          |   objects*:  |              |
|                  |                          |   figure,    |              |
|                  |                          |   "All of the|              |
|                  |                          |   figures for|              |
|                  |                          |   this page  |              |
|                  |                          |   are        |              |
|                  |                          |   static."   |              |
+------------------+--------------------------+--------------+--------------+
| Axes_            | Subplots within Figure.  | - *For       | - "The axes  |
|                  | Contains plot elements   |   Matplotlib |   methods    |
|                  | and is responsible for   |   objects*:  |   transform  |
|                  | plotting and configuring |   Axes, "An  |   the data." |
|                  | additional details.      |   Axes is a  | - "Each Axes_|
|                  |                          |   subplot    |   is specific|
|                  |                          |   within the |   to a       |
|                  |                          |   Figure."   |   Figure."   |
|                  |                          | - *Referring | - "All of the|
|                  |                          |   to class*: |   Axes are   |
|                  |                          |   Axes_ ,    |   compatible |
|                  |                          |   "Each Axes_|   with other |
|                  |                          |   is specific|   libraries."|
|                  |                          |   to one     |              |
|                  |                          |   Figure."   |              |
|                  |                          | - *General   |              |
|                  |                          |   objects*:  |              |
|                  |                          |   axes, "Both|              |
|                  |                          |   explicit   |              |
|                  |                          |   and        |              |
|                  |                          |   implicit   |              |
|                  |                          |   strategies |              |
|                  |                          |   contain    |              |
|                  |                          |   multiple   |              |
|                  |                          |   types of   |              |
|                  |                          |   data       |              |
|                  |                          |   visuals    |              |
|                  |                          |   on a single|              |
|                  |                          |   axes."     |              |
+------------------+--------------------------+--------------+--------------+
| Artist_          | Broad variety of         | - *For       | - "Configure |
|                  | Matplotlib objects that  |   Matplotlib |   the legend |
|                  | display visuals.         |   objects*:  |   artist with|
|                  |                          |   Artist,    |   its        |
|                  |                          |   "Artists   |   respective |
|                  |                          |   display    |   method."   |
|                  |                          |   visuals and| - "There are |
|                  |                          |   are the    |   many       |
|                  |                          |   visible    |   Artists_ in|
|                  |                          |   elements   |   this       |
|                  |                          |   when the   |   graph."    |
|                  |                          |   rendering  | - "Some      |
|                  |                          |   a Figure." |   Artists    |
|                  |                          | - *Referring |   have an    |
|                  |                          |   to class*: |   overlapping|
|                  |                          |   Artist_ ,  |   naming     |
|                  |                          |   "Each      |   convention |
|                  |                          |   Artist_ has|   in both    |
|                  |                          |   respective |   explicit   |
|                  |                          |   methods and|   and        |
|                  |                          |   functions."|   implicit   |
|                  |                          | - *General   |   uses."     |
|                  |                          |   objects*:  |              |
|                  |                          |   artist,    |              |
|                  |                          |   "The       |              |
|                  |                          |   artists in |              |
|                  |                          |   this table |              |
|                  |                          |   generate   |              |
|                  |                          |   visuals."  |              |
+------------------+--------------------------+--------------+--------------+
| Axis_            | Human-readable single    | - One Axis   | - One Axis   |
|                  | dimensional object       |   object     | - One axes   |
|                  | of reference marks       | - Four axis  | - Four Axises|
|                  | containing ticks, tick   |   objects    | - 32 Axes    |
|                  | labels, spines, and      | - 32 Axis    |              |
|                  | edges.                   |   objects    |              |
+------------------+--------------------------+--------------+--------------+
| Explicit,        | Explicit approach of     | - Explicit   | - object     |
| Object Oriented  | programing in Matplotlib.| - explicit   |   oriented   |
| Programming (OOP)|                          | - OOP        | - OO-style   |
+------------------+--------------------------+--------------+--------------+
| Implicit,        | Implicit approach of     | - Implicit   | - MATLAB like|
| ``pyplot``       | programming in Matplotlib| - implicit   | - Pyplot     |
|                  | with ``pyplot`` module.  | - ``pyplot`` | - pyplot     |
|                  |                          |              |   interface  |
+------------------+--------------------------+--------------+--------------+

.. _Figure: :class:`~matplotlib.figure.Figure`
.. _Axes: :class:`~matplotlib.axes.Axes`
.. _Artist: :class:`~matplotlib.artist.Artist`
.. _Axis: :class:`matplotlib.axis.Axis`


Grammar
-------

Subject
^^^^^^^
Use second-person imperative sentences for directed instructions specifying an
action. Second-person pronouns are for individual-specific contexts and
possessive reference.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| Install Matplotlib from the source | You can install Matplotlib from the|
| directory using the Python ``pip`` | source directory. You can find     |
| installer program. Depending on    | additional support if you are      |
| your operating system, you may need| having trouble with your           |
| additional support.                | installation.                      |
+------------------------------------+------------------------------------+

Tense
^^^^^
Use present simple tense for explanations. Avoid future tense and other modal
or auxiliary verbs when possible.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| The fundamental ideas behind       | Matplotlib will take data and      |
| Matplotlib for visualization       | transform it through functions and |
| involve taking data and            | methods. They can generate many    |
| transforming it through functions  | kinds of visuals. These will be the|
| and methods.                       | fundamentals for using Matplotlib. |
+------------------------------------+------------------------------------+

Voice
^^^^^
Write in active sentences. Passive voice is best for situations or conditions
related to warning prompts.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| The function ``plot`` generates the| The graph is generated by the      |
| graph.                             | ``plot`` function.                 |
+------------------------------------+------------------------------------+
| An error message is returned by the| You will see an error message from |
| function if there are no arguments.| the function if there are no       |
|                                    | arguments.                         |
+------------------------------------+------------------------------------+

Sentence structure
^^^^^^^^^^^^^^^^^^
Write with short sentences using Subject-Verb-Object order regularly. Limit
coordinating conjunctions in sentences. Avoid pronoun references and
subordinating conjunctive phrases.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| The ``pyplot`` module in Matplotlib| The ``pyplot`` module in Matplotlib|
| is a collection of functions. These| is a collection of functions which |
| functions create, manage, and      | create, manage, and manipulate the |
| manipulate the current Figure and  | current Figure and plotting area.  |
| plotting area.                     |                                    |
+------------------------------------+------------------------------------+
| The ``plot`` function plots data   | The ``plot`` function plots data   |
| to the respective Axes. The Axes   | within its respective Axes for its |
| corresponds to the respective      | respective Figure.                 |
| Figure.                            |                                    |
+------------------------------------+------------------------------------+
| The implicit approach is a         | Users that wish to have convenient |
| convenient shortcut for            | shortcuts for generating plots use |
| generating simple plots.           | the implicit approach.             |
+------------------------------------+------------------------------------+


Formatting
==========

The following guidelines specify how to incorporate code and use appropriate
formatting for Matplotlib documentation.

Code
----

Matplotlib is a Python library and follows the same standards for
documentation.

Comments
^^^^^^^^
Examples of Python code have comments before or on the same line.

+---------------------------------------+---------------------------------+
| Correct                               | Incorrect                       |
+=======================================+=================================+
| ::                                    | ::                              |
|                                       |                                 |
|    # Data                             |    years = [2006, 2007, 2008]   |
|    years = [2006, 2007, 2008]         |    # Data                       |
+---------------------------------------+                                 |
| ::                                    |                                 |
|                                       |                                 |
|    years = [2006, 2007, 2008]  # Data |                                 |
+---------------------------------------+---------------------------------+

Outputs
^^^^^^^
When generating visuals with Matplotlib using ``.py`` files in examples,
display the visual with `matplotlib.pyplot.show` to display the visual.
Keep the documentation clear of Python output lines.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| ::                                 | ::                                 |
|                                    |                                    |
|    plt.plot([1, 2, 3], [1, 2, 3])  |    plt.plot([1, 2, 3], [1, 2, 3])  |
|    plt.show()                      |                                    |
+------------------------------------+------------------------------------+
| ::                                 | ::                                 |
|                                    |                                    |
|    fig, ax = plt.subplots()        |    fig, ax = plt.subplots()        |
|    ax.plot([1, 2, 3], [1, 2, 3])   |    ax.plot([1, 2, 3], [1, 2, 3])   |
|    fig.show()                      |                                    |
+------------------------------------+------------------------------------+

reStructuredText
----------------

Matplotlib uses reStructuredText Markup for documentation. Sphinx helps to
transform these documents into appropriate formats for accessibility and
visibility.

- `reStructuredText Specifications <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html>`_
- `Quick Reference Document <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_


Lists
^^^^^
Bulleted lists are for items that do not require sequencing. Numbered lists are
exclusively for performing actions in a determined order.

+------------------------------------+------------------------------------+
| Correct                            | Incorrect                          |
+====================================+====================================+
| The example uses three graphs.     | The example uses three graphs.     |
+------------------------------------+------------------------------------+
| - Bar                              | 1. Bar                             |
| - Line                             | 2. Line                            |
| - Pie                              | 3. Pie                             |
+------------------------------------+------------------------------------+
| These four steps help to get       | The following steps are important  |
| started using Matplotlib.          | to get started using Matplotlib.   |
+------------------------------------+------------------------------------+
|  1. Import the Matplotlib library. |  - Import the Matplotlib library.  |
|  2. Import the necessary modules.  |  - Import the necessary modules.   |
|  3. Set and assign data to work on.|  - Set and assign data to work on. |
|  4. Transform data with methods and|  - Transform data with methods and |
|     functions.                     |    functions.                      |
+------------------------------------+------------------------------------+

Tables
^^^^^^
Use ASCII tables with reStructuredText standards in organizing content. 
Markdown tables and the csv-table directive are not accepted.

+--------------------------------+----------------------------------------+
| Correct                        | Incorrect                              |
+================================+========================================+
| +----------+----------+        | ::                                     |
| | Correct  | Incorrect|        |                                        |
| +==========+==========+        |     | Correct | Incorrect |            |
| | OK       | Not OK   |        |     | ------- | --------- |            |
| +----------+----------+        |     | OK      | Not OK    |            |
|                                |                                        |
+--------------------------------+----------------------------------------+
| ::                             | ::                                     |
|                                |                                        |
|     +----------+----------+    |     .. csv-table::                     |
|     | Correct  | Incorrect|    |        :header: "correct", "incorrect" |
|     +==========+==========+    |        :widths: 10, 10                 |
|     | OK       | Not OK   |    |                                        |
|     +----------+----------+    |        "OK   ", "Not OK"               |
|                                |                                        |
+--------------------------------+                                        |
| ::                             |                                        |
|                                |                                        |
|     ===========  ===========   |                                        |
|       Correct     Incorrect    |                                        |
|     ===========  ===========   |                                        |
|     OK           Not OK        |                                        |
|     ===========  ===========   |                                        |
|                                |                                        |
+--------------------------------+----------------------------------------+


Additional resources
====================
This style guide is not a comprehensive standard. For a more thorough
reference of how to contribute to documentation, see the links below. These
resources contain common best practices for writing documentation.  

* `Python Developer's Guide <https://devguide.python.org/documenting/#documenting-python>`_
* `Google Developer Style Guide <https://developers.google.com/style>`_
* `IBM Style Guide <https://www.ibm.com/developerworks/library/styleguidelines/>`_
* `Red Hat Style Guide <https://stylepedia.net/style/#grammar>`_

