
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
reliability and consistency in documentation. They are case-sensitive and are
not interchangeable.

+------------------+--------------------------+--------------+--------------+
| Term             | Description              | Correct      | Incorrect    |
+==================+==========================+==============+==============+
| Figure_          | Matplotlib working space | - One Figure | - One figure |
|                  | for programming.         | - 11 Figures | - 11 figures |
+------------------+--------------------------+--------------+--------------+
| Axes_            | Subplots within Figure.  | - One Axes   | - One axes   |
|                  | Contains plot elements   | - Four Axes  | - Four Axeses|
|                  | and is responsible for   | - 32 Axes    | - 32 Axii    |
|                  | plotting and configuring |              |              |
|                  | additional details.      |              |              |
+------------------+--------------------------+--------------+--------------+
| Artist_          | Broad variety of         | - One Artist | - One artist |
|                  | Matplotlib objects that  | - Two Artists| - Two artists|
|                  | display visuals.         |              |              |
+------------------+--------------------------+--------------+--------------+
| Axis_            | Human-readable single    | - One Axis   | - One Axis   |
|                  | dimensional object       |   object     | - One axis   |
|                  | of reference marks       | - Four Axis  | - Four Axises|
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

