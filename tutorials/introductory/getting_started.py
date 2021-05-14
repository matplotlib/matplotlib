"""

.. _getting_started:

***************
Getting Started
***************

This tutorial covers basic usage patterns and best practices to help get
started using Matplotlib.
"""

##############################################################################
#
# Introduction
# ============
#
# Matplotlib is a Python library providing tools for users to create
# visualizations with data.
#
# The library is accessible through a variety of operating systems and
# programming environments. The fundamental ideas behind Matplotlib for
# visualizations involve taking data and transforming it through functions and
# methods. This process occurs internally and is not user-facing.
#
# There are two main ways of producing graphs with Matplotlib, explicit and
# implicit. Explicit code, using Object Oriented Programming (OOP), and
# implicit code, using ``pyplot``, are the foundation for creating and
# manipulating data into visualizations.
#
# +------------------------------------+------------------------------------+
# | Explicit, Object Oriented          | Implicit, ``pyplot``               |
# | Programming (OOP)                  |                                    |
# +====================================+====================================+
# | Users explicitly create and manage | The Matplotlib library implicitly  |
# | all plot elements.                 | manages Figure and Axes.           |
# +------------------------------------+------------------------------------+
# | Useful for repeated code use,      | Helpful for quickly graphing data  |
# | generalization, robust             | when using interactive             |
# | configurations of visuals.         | environments.                      |
# +------------------------------------+------------------------------------+
# | Recommended to new users for       | Most useful for users coming from  |
# | learning fundamentals.             | MATLAB. Users already familiar with|
# |                                    | Matplotlib also benefit from using |
# |                                    | ``pyplot`` for convenient          |
# |                                    | shortcuts.                         |
# +------------------------------------+------------------------------------+
#
# Explicit programming helps users generalize code and is useful for repeated
# uses or larger projects. This is also a more robust way of controlling
# customizations for visualizations. Users looking to have control over every
# part of the graph call methods on each item. Most users benefit using
# explicit programming for regular Matplotlib use as the user manages each
# element of building a graph.
#
# Implicit programming with ``pyplot`` is simpler. It is helpful for basic
# plots and for interactive environments, such as Jupyter Notebooks. Users
# familiar with MATLAB or wishing to have Matplotlib create and manage parts of
# the visualization in state-based programming benefit from using ``pyplot``.
# Using implicit programming acts as a convenient shortcut for generating
# visualizations. New users to Matplotlib may experience difficulty
# understanding how elements of a visualization work together when using the
# implicit approach.
#
# Examples
# --------
#
# The table below depicts the two alternative approaches to plotting a
# simple graph. The image following the table is the visualization of the
# programming.
#
# +------------------------------------+------------------------------------+
# | Explicit, Object Oriented          | Implicit, ``pyplot``               |
# | Programming (OOP)                  |                                    |
# +====================================+====================================+
# | ::                                 | ::                                 |
# |                                    |                                    |
# |     fig, ax = plt.subplots()       |    plt.plot([1, 2, 3], [1, 2, 3])  |
# |     ax.plot([1, 2, 3], [1, 2, 3])  |                                    |
# |                                    |                                    |
# +------------------------------------+------------------------------------+
# | `.pyplot.subplots` generates a     | :mod:`matplotlib.pyplot` creates   |
# | `~.figure.Figure` and one or       | implicit Figure and Axes if        |
# | more `~.axes.Axes` explicitly.     | there are no pre-existing          |
# | `.Axes.plot` plots the data.       | elements and `.pyplot.plot` plots  |
# |                                    | the data. This also plots over any |
# |                                    | existing Figure if applicable.     |
# +------------------------------------+------------------------------------+
# | .. plot::                          | .. plot::                          |
# |                                    |                                    |
# |     fig, ax = plt.subplots()       |     plt.plot([1, 2, 3], [1, 2, 3]) |
# |     ax.plot([1, 2, 3], [1, 2, 3])  |                                    |
# |                                    |                                    |
# +------------------------------------+------------------------------------+
#
# .. note::
#
#     The example graphs are identical for both explicit and implicit code.
#
# Requirements
# ============
#
# Matplotlib is a Python library and an installed version of *Python 3.6 or
# higher* is required. Depending on your operating system, Python may already
# be installed on your machine.
#
# Installing Matplotlib is required in order to generate graphs with the
# library. Install Matplotlib for your own development environment manually or
# use a third-party package distribution.
#
# The `Installation Guide <https://matplotlib.org/users/installing.html>`_
# page contains more information about install methods and resources for
# third-party package distributions.
#
# .. seealso::
#
#     To contribute to the Matplotlib community, check
#     :ref:`developers-guide-index`
#     for details about working with the latest sources.
#
# Interactive environments
# ------------------------
#
# The Matplotlib community suggests using `IPython <https://ipython.org/>`_
# through `Jupyter <https://jupyter.org/index.html>`_ as the primary
# interactive environment.
#
# Plotting
# ========
#
# The common convention for preparing to plot data involves importing the
# Matplotlib library module ``pyplot`` with the abbreviation ``plt`` for
# convenience. Both explicit and implicit programming require the module.
#
# The other library imports are for support. Explanations on their purposes
# are included below.

import matplotlib.pyplot as plt

##############################################################################
#
# - The ``pyplot`` module in Matplotlib is a collection of functions. The
#   module's functions create, manage, and manipulate the current Figure and
#   the plotting area. The ``plt`` abbreviation is the standard shortcut.
#

import numpy as np

from functools import partial

##############################################################################
#
# - `NumPy <https://numpy.org/doc/stable/index.html#>`_ is a common scientific
#   Python library that benefits users working with numerical data.
# - The ``functools`` module helps manage functions that act on or return
#   other functions. The `Pie Chart Examples`_ section note contains more
#   information about the purpose of this module.
#
#
#
# Two Approaches for Creating Graphs
# ----------------------------------
#
# The two strategies, explicit and implicit, both involve using the ``pyplot``
# module. However, they differ in how users interact with the data in the
# transformation process. The `Introduction`_ and `Examples`_ sections above
# provide key differences.
#
# +------------------------------------+------------------------------------+
# | Explicit                           | Implicit                           |
# +====================================+====================================+
# | - Code has explicit references to  | - The programming is designed to   |
# |   objects. Users manage objects for|   remember preceding events or     |
# |   the specific Figure and Axes and |   interactions. This preserved     |
# |   call on methods for manipulating |   state allows Matplotlib to       |
# |   data.                            |   automatically manage a Figure and|
# | - Object Oriented Programming      |   Axes.                            |
# |   allows for robust control and is | - The module ``pyplot`` operates   |
# |   useful for generalized code.     |   similarly to MATLAB and is       |
# |                                    |   convenient for interactive       |
# |                                    |   environments.                    |
# +------------------------------------+------------------------------------+
#
# .. note::
#
#     The Matplotlib community does not recommend interchanging explicit and
#     implicit strategies. When using one as standard, all code following
#     the same strategy reduces troubleshooting issues. Switching back and
#     forth between explicit and implicit programming may yield errors.
#
# For other techniques of creating plots with Matplotlib, refer to
# :ref:`user_interfaces`.
#
# Data
# ----
#
# The Matplotlib library manages data in the form of iterables and/or
# sequenced items. These also take the form of NumPy arrays like
# `numpy.array` or `numpy.ma.masked_array`. All plotting functions take these
# data structures.
#

# Sample Data for Personal Financial Tracking in 2009 & 2010

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
income = [950, 950, 950, 950, 950, 950,
          950, 950, 950, 950, 950, 950]
chk_acct_09 = [1250, 1325, 1200, 1220, 1100, 1055,
               1255, 1090, 1190, 1205, 1205, 1180]
svg_acct_09 = [1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550]
chk_acct_10 = [1180, 1270, 1280, 1280, 1260, 1140,
               1270, 1160, 1120, 1250, 1270, 1160]
svg_acct_10 = [1550, 1600, 1650, 1700, 1750, 1800,
               1850, 1900, 1950, 2000, 2050, 2100]

##############################################################################
#
# .. note::
#
#    Other containers, such as data objects from various libraries, may not
#    work as intended.
#
# Explicit and implicit examples of a basic line plot are below. Both of the
# following plots are identical. Each uses a different approach to graph the
# data. The results do not change for either approach when using the same data
# points.
#
# Explicit: Object Oriented Programming (OOP)
# --------------------------------------------
#
# Explicit programming for Matplotlib involves calling the function
# `matplotlib.pyplot.subplots` in the ``pyplot`` module once. This returns a
# group of an explicit Figure and Axes to be unpacked as part of variable
# assignment. More than one Axes is configurable; however, each Axes only
# corresponds to a single Figure.
#
# Each Axes has its own methods to graph data. In addition, each Axes
# also uses separate methods to create and manage objects within a Figure.
# These methods are different from those of the implicit programming approach.

# Explicit programming with OOP

# Assigning sample data to variables.
x = months
y1 = income
y2 = chk_acct_09
y3 = svg_acct_09
y4 = chk_acct_10
y5 = svg_acct_10

# Explicit Figure & Axes unpacked separately with module.
# Conventional object abbreviations are `fig` and `ax`, respectively.
fig, ax = plt.subplots()

# Single explicit Axes graphs multiple data points.
ax.plot(x, y1, label='Income')
ax.plot(x, y2, label='Checking Account')
ax.plot(x, y3, label='Savings Account')

# Explicit Axes use separate methods to manage parts of Figure.
ax.set_xlabel('Month')
ax.set_ylabel('USD')
ax.set_title('Personal Financial Tracking from 2009')
ax.legend()

# The pyplot module displays the Figure.
plt.show()

##############################################################################
#
# The module ``pyplot`` for the explicit example uses a function that returns
# the Figure and Axes. This convention uses ``plt.subplots()``. It defaults
# to one Figure, ``fig``, and one Axes, ``ax``. The variable names are common
# shorthand terms and any naming conventions also work.
#
# The `Configuration`_ section below contains additional information about
# manipulating visuals, multiple visualizations, and other modifications.
#
# Using explicit programming allows for ``fig`` and ``ax`` to use separate
# methods to manage objects within the visualization. Specific Figures and
# Axes manage data components with their own respective methods.
#
#
# Implicit: ``pyplot``
# --------------------
#
# Implicit programming for Matplotlib centers around using the ``pyplot``
# module. The module implicitly generates the Figure and Axes. Methods and
# functions within the module take incoming data as arguments. Additional parts
# of the Figure are also available through the module methods.

# Implicit programming with pyplot

# Previous variables are still referenced.

# Module plots multiple data points on implicitly generated Axes.
plt.plot(x, y1, label='Income')
plt.plot(x, y2, label='Checking Account')
plt.plot(x, y3, label='Savings Account')

# Module methods generate parts of Figure.
plt.xlabel('Month')
plt.ylabel('USD')
plt.title("Personal Financial Tracking from 2009")
plt.legend()

# The module displays the Figure.
plt.show()

##############################################################################
#
# In the example above, the ``pyplot`` module contains its own functions of
# actionable tasks for the data. The ``plt.plot`` plots data as a line graph
# with various keyword arguments as customizable options. The module also
# includes other methods for generating parts of the visualization. These parts
# use different methods from the explicit approach.
#
# .. note::
#
#    The names and spelling for methods may be similar for both explicit and
#    implicit approaches. Errors may occur when using the wrong corresponding
#    method. Confirm with the documentation API of `~.axes.Axes` for explicit
#    and :mod:`matplotlib.pyplot` for implicit or other respective method
#    names.
#
# Configuration
# =============
#
# There are two main parts to building a visualization with Matplotlib, the
# ``Figure`` and the ``Axes``.
#
# Components of Matplotlib Figure
# -------------------------------
#
# The image below depicts each visible element of a Matplotlib graph. The
# graphic uses Matplotlib to display and highlight each individual part of the
# visualization. To view source code for the image, see
# :ref:`sphx_glr_gallery_showcase_anatomy.py`.
#
#
# .. image:: ../../_static/anatomy.png
#
#
# .. note::
#
#     ``Figure`` and ``Axes`` identify empty regions of the diagram;
#     however, these elements are foundational in operation. The example below
#     illustrates an empty Figure and respective Axes. Matplotlib also
#     automatically generates certain Artists for the visualization even
#     without assigned data.
#

# Explicit Figure and Axes unpacked from module function.
# No data transformed for visualizations.
fig, ax = plt.subplots()

# Module displays empty Figure and Axes.
plt.show()

##############################################################################
#
# :class:`~matplotlib.figure.Figure`
#
# The Figure is the working space for the programming. All visible
# objects on a graph are located within the Figure.
#
# :class:`~matplotlib.axes.Axes`
#
# Axes are subplots within the Figure. They contain Matplotlib objects and
# are responsible for plotting and configuring additional details. Each
# Figure can contain multiple Axes, but each Axes is specific to one
# Figure.
#
# In a Figure, each Axes contains any number of plot elements. Axes are
# configurable for more than one type of visualization of data. From the
# `Plotting`_ section above, the Axes in both explicit and implicit strategies
# contain multiple types of visualizations of data on a single Axes.
#
# Each of these types are specific to the Axes they are in. In the example, the
# two plots each have one Axes. These Axes each have multiple plot lines. The
# lines as objects are not shared between the two plots even though the data is
# shared.
#
# Matplotlib Axes also integrate with other Python libraries. In Axes-based
# interfaces, other libraries take an Axes object as input. Libraries such as
# `pandas` and `Seaborn <https://seaborn.pydata.org>`_ act on specific Axes.
#
# Other Components
# ^^^^^^^^^^^^^^^^
#
# :class:`~matplotlib.artist.Artist`
#
# Artists are a broad variety of Matplotlib objects. They display visuals and
# are the visible elements when the Figure is rendered. They correspond to a
# specific Axes and cannot be shared or transferred. In Matplotlib programming,
# all objects for display are Artists.
#
# .. note::
#
#   Axes and Axis are not synonymous. Axis refers to
#   :class:`~matplotlib.axis.Axis`, a separate Matplotlib object.
#
# Manipulating Artists
# --------------------
#
# With simple plots, Matplotlib automatically generates the basic plot elements
# of a graph. For more control over the visual, use Artists and methods.
#
# Matplotlib generates additional visual elements as Artists in the form of
# objects. As Artists, each has respective methods and functions. Explicit and
# implicit approaches use different methods and are not interchangeable.
#
# +-----------------------+--------------------------+------------------------+
# | Artist                | Explicit                 | Implicit               |
# +=======================+==========================+========================+
# | Visible elements from | Each specific Axes has   | The ``pyplot`` module  |
# | rendered Figure.      | its own method for       | manages Artists based  |
# |                       | Artists.                 | on most recent Figure  |
# |                       |                          | or Axes.               |
# +-----------------------+--------------------------+------------------------+
#
# The table below compares common formatter Artists and their different
# methods. These Artists label and identify parts of a visualization.
#
# The term ``ax`` refers to an assigned variable for a specific Axes. Using
# explicit programming may require additional tasks of setting objects prior
# to assigning labels. Whereas with implicit programming, the module manages
# those tasks without specification.
#
# +-----------------------+--------------------------+------------------------+
# | Artist                | Explicit                 | Implicit               |
# +=======================+==========================+========================+
# | X-Axis labels         | ``ax.set_xticks()``      | ``plt.xticks()``       |
# |                       | ``ax.set_xticklabels()`` |                        |
# +-----------------------+--------------------------+------------------------+
# | Y-Axis labels         | ``ax.set_yticks()``      | ``plt.yticks()``       |
# |                       | ``ax.set_yticklabels()`` |                        |
# +-----------------------+--------------------------+------------------------+
# | Title (Axes)          | ``ax.set_title()``       | ``plt.title()``        |
# +-----------------------+--------------------------+------------------------+
#
# The following table represents common Artists for transforming data. The
# Artists in this table generate data visualizations from transformations.
# These methods often overlap in naming conventions and make use of identical
# keyword arguments and other parameters.
#
# +-----------------------+--------------------------+------------------------+
# | Artist                | Explicit                 | Implicit               |
# +=======================+==========================+========================+
# | Plot                  | ``ax.plot()``            | ``plt.plot()``         |
# +-----------------------+--------------------------+------------------------+
# | Pie                   | ``ax.pie()``             | ``plt.pie()``          |
# +-----------------------+--------------------------+------------------------+
# | Legend (Axes)         | ``ax.legend()``          | ``plt.legend()``       |
# +-----------------------+--------------------------+------------------------+
#
# Supplemental Resources
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Customizations with robust options have their own guides and tutorials. The
# topics below include common in-depth documents for additional support.
#
# +------------------------------+--------------------------------------------+
# | Topic                        | Tutorial                                   |
# +==============================+============================================+
# | :ref:`tutorials-introductory`| :doc:`/tutorials/introductory/customizing` |
# +------------------------------+--------------------------------------------+
# | :ref:`tutorials-intermediate`| :doc:`/tutorials/intermediate/legend_guide`|
# +------------------------------+--------------------------------------------+
# | :ref:`tutorials-colors`      | :doc:`/tutorials/colors/colors`            |
# |                              +--------------------------------------------+
# |                              | :doc:`/tutorials/colors/colormaps`         |
# +------------------------------+--------------------------------------------+
# | :ref:`tutorials-text`        | :doc:`/tutorials/text/text_intro`          |
# |                              +--------------------------------------------+
# |                              | :doc:`/tutorials/text/text_props`          |
# |                              +--------------------------------------------+
# |                              | :doc:`/tutorials/text/annotations`         |
# +------------------------------+--------------------------------------------+
#
# For complete information about available methods for creating new Artists,
# refer to the table below.
#
# +------------------------------------+------------------------------------+
# | Explicit                           | Implicit                           |
# +====================================+====================================+
# | :class:`matplotlib.axes.Axes`      | :mod:`matplotlib.pyplot`           |
# +------------------------------------+------------------------------------+
#
#
# Pie Chart Examples
# ------------------
#
# Matplotlib pie charts create wedges based on data. They manipulate the size
# of the Artists based on the ratio of the wedge to the sum of the data. The
# ``.pie()`` method is similar in both explicit and implicit approaches.
#
# The code below illustrates various levels of configuration in keyword
# arguments as well as Artist methods for both explicit and implicit
# programming.
#

# Sample data for monthly spending averages.

# Data points correspond to wedge size as a ratio of total sum.
# Matplotlib methods calculate these values automatically based on input.
budget = [475, 300, 125, 50]

# Lists of strings contribute to labeling corresponding data.
descriptions = ['Shared house\nin Philadelphia',
                'Dog costs, phone,\nutilities',
                'Groceries\n& takeout',
                'Treasury bonds']
categories = ['Rent', 'Bills', 'Food', 'Savings']

# Hex color codes determine respective wedge color.
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# List of floats represents percentage of radius to separate from center.
explode = [0, 0.1, 0.15, 0.35]

# This function operates in conjunction with the functools partial function
# for formatting labels in wedges.


def autopct_format(percent, group):
    """
    Takes percent equivalent and calculates original value from data.
    Returns string of value new line above percentage.

    Parameters
    ----------
    percent : float
        Number as percentage equivalent
    group : array
        Collection of values

    Returns
    -------
    formatted : fstring
        Formatted string with symbols, spacing, and line breaks
    """
    value = int(percent/100.*np.sum(group))
    formatted = f'${value:<4}\n{percent:1.1f}%'
    return formatted


##############################################################################
#
# Basic
# ^^^^^
#
# The following two plots are identical. Both the explicit and implicit
# approaches generate the exact same plot when using the same variables.
#
# Review `matplotlib.axes.Axes.pie` and `matplotlib.pyplot.pie` for more
# information about the APIs for explicit and implicit, respectively.

# Explicit

fig, ax = plt.subplots()

ax.pie(budget, colors=colors, labels=categories)
ax.legend()
ax.set_title('Average Monthly Income Expenses')
ax.axis('equal')  # The axis method sets the aspect ratio as equal.

plt.show()

##############################################################################
#
#

# Implicit

plt.pie(budget, colors=colors, labels=categories)
plt.legend()
plt.title('Average Monthly Income Expenses')
plt.axis('equal')  # The pyplot module has identical method for aspect ratio.

plt.show()

##############################################################################
#
# .. note::
#
#   There are minor differences in the method names. Overall, each method
#   performs the same action through the different approaches.
#
# These pie charts are simple and do not have distinguishing information.
# Keyword arguments and Artists add the ability to implement more ways of
# displaying content.
#
# Additional Customization
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Many methods contain optional keyword arguments for further configuration.
# In the examples for explicit programming below, there are values and
# functions in keyword arguments for formatting the Artists. These changes also
# apply to implicit programming, though with varying method names.
#
# The pie chart below adds configurations with keyword arguments for
# ``explode``, ``autopct``, ``startangle``, and ``shadow``. These keyword
# arguments help to define the display of Artists.

# Explicit

fig, ax = plt.subplots()

# The explode keyword argument uses explode variable data to separate
# respective wedges from center.
# The autopct keyword argument takes formatting strings and functions to
# generate text within each wedge. '%1.1f%%' is the string formatter.
# The startangle keyword argument changes where first wedge spans. Angles start
# at 0 degrees on X-axis and move counterclockwise.
# The shadow keyword argument toggles a shadow on the visual.
ax.pie(budget,
       colors=colors,
       labels=categories,
       explode=explode,
       autopct='%1.1f%%',
       startangle=-80,
       shadow=True)

ax.legend()
ax.set_title('Average Monthly Income Expenses')
ax.axis('equal')

plt.show()

##############################################################################
#
# The following pie chart has additional keyword arguments to further
# customize the visual. Also, the ``legend`` as an Artist has parameters that
# enable more specification for the information displayed. For more, see the
# :doc:`/tutorials/intermediate/legend_guide`.

# Explicit

fig, ax = plt.subplots()

# Descriptions now act as text labels for wedges. This removes redundant
# information from previous pie chart.
# The autopct keyword argument calls a function as well. The functools partial
# function returns a formatted string. See Note below for more.
# The pctdistance keyword argument places autopct Artist at a location using
# float as percentage of radius.
# The labeldistance keyword argument specifies float as percentage of radius to
# place labels.
# The wedgeprops keyword argument also takes dictionaries to pass to Artists.
# The float for width sets wedge size as percentage of radius starting from
# outer edge.
budget_pie = ax.pie(budget,
                    colors=colors,
                    labels=descriptions,
                    explode=explode,
                    autopct=partial(autopct_format, group=budget),
                    startangle=45,
                    pctdistance=0.85,
                    labeldistance=1.125,
                    wedgeprops=dict(width=0.3),
                    shadow=True)

# The pie() method unpacks into three Artist objects. The Artists wedges,
# texts, and autotexts have their own methods for addtional customization.
wedges, texts, autotexts = budget_pie

# The unpacked wedges variable serve as handles for legend.
# Info from categories correspond to respective wedge instead of redundant
# labeling from previous pie chart.
# Legend has title keyword argument.
# Keyword argument bbox_to_anchor with loc places legend at specific point.
# Tuple floats are coordinates for Figure as row and column of Axes.
# Keyword argument loc works with bbox_to_anchor to determine part of legend
# for placement. Without bbox_to_anchor, Matplotlib automatically manages
# coordinates in relation to parameters of loc.
ax.legend(wedges,
          categories,
          title='Categories',
          bbox_to_anchor=(0.125, 0.5),
          loc='center right')

ax.set_title('Average Monthly Income Expenses')
ax.axis('equal')

# The Figure method tight_layout() adjusts spacing between all Artists to
# maximize visiblity on the Figure. This method also contains various
# parameters for configuration.
fig.tight_layout()

plt.show()

##############################################################################
#
# .. note::
#
#   The ``partial`` function in functools works as a callable for simplifying
#   a function's arguments. In the ``autopct`` keyword argument, only one
#   argument is provided, the data acting as a percentage equivalent. The
#   ``autopct_format`` function requires two arguments, so ``partial`` takes
#   the argument for ``group`` and sets it to ``budget``. This smaller
#   signature object then behaves as the same function with one fewer argument.
#   For details about the functools module, see
#   `functools
#   <https://docs.python.org/3/library/functools.html#module-functools>`_.
#
# Multiple Graphs within a Figure
# -------------------------------
#
# For multiple graphs using a single Figure, explicit and implicit approaches
# use a similar convention for mapping out multiple Axes. Matplotlib manages
# more than one Axes in a two-dimensional matrix. They are arranged by row
# amount and then by column amount.
#
# Implicit coding uses a separate method with a similar name. The method
# ``plt.subplot`` also includes a third argument to represent the specific
# Axes involved.
#
# When looking for more complex solutions to multiple graphs within a Figure,
# use the :class:`matplotlib.gridspec.GridSpec` module to organize the layout.
#
# Explicit
# ^^^^^^^^

# Explicit with OOP

# Figure and two Axes unpacked from matrix as row (1) & column (2).
# Keyword arguments provide additional details of sharing Y-Axis, Figure size
# and layout formatting.
fig, (ax1, ax2) = plt.subplots(1, 2,
                               sharey='row',
                               figsize=[8, 4],
                               constrained_layout=True)

# Explicit Figure object has separate method for title.
fig.suptitle('Personal Financial Tracking \'09 - \'10')

# First explicit Axes object plots data with additional keyword arguments.
ax1.plot(x, y1, label='Income')
ax1.plot(x, y2, label='Checking')
ax1.plot(x, y3, color='green', label='Savings')

# First explicit Axes object uses separate methods for ticks on X-Axis,
# title, and legend. Keyword arguments are for additional configurations.
ax1.set_xticks(months)
ax1.set_xticklabels(months, rotation=270)
ax1.set_title('2009', fontsize='small')
ax1.legend(loc='upper left')

# Explicit second Axes object plots data similarly to first explicit Axes.
ax2.plot(x, y1, label='Income')
ax2.plot(x, y4, label='Checking')
ax2.plot(x, y5, color='green', label='Savings')

# Explicit second Axes object has separate methods as well.
ax2.set_xticks(months)
ax2.set_xticklabels(months, rotation=270)
ax2.set_title('2010', fontsize='small')

# The pyplot module displays Figure.
plt.show()

##############################################################################
#
# The explicit example above also uses two Axes to graph the data. However,
# the explicit approach refers to an explicitly generated Axes after creating
# both the Figure and Axes.
#
# In the unpacking process, multiple Axes are assigned to a single variable.
# To reference a specific Axes, indexing the location of the respective Axes
# as a matrix through the single variable works as well.
#
# The code below demonstrates indexing multiple Axes::
#
#   fig, ax = plt.subplots(2, 2)
#
#   ax[0,0].bar([1, 2, 3], [1, 2, 3])
#   ax[0,1].plot([3, 2, 1], [3, 2, 1])
#   ax[1,0].hist(hist_data)
#   ax[1,1].imshow([[1, 2], [2, 1]])
#
#
# The method `matplotlib.figure.Figure.subplot_mosaic` also generates Axes in
# a layout with contextual names. The link contains more info for using the
# method.
#
# See code example below::
#
#   fig = plt.figure()
#   ax_dict = fig.subplot_mosaic([['bar', 'plot'],
#                                ['hist', 'image']])
#
#   ax_dict['bar'].bar([1, 2, 3], [1, 2, 3])
#   ax_dict['plot'].plot([3, 2, 1], [3, 2, 1])
#   ax_dict['hist'].hist(hist_data)
#   ax_dict['image'].imshow([[1, 2], [2, 1]])
#
# Implicit
# ^^^^^^^^
#
# There are limitations for customizing the implicit approach without
# referencing specific Axes and Artists within the Figure. For more advanced
# configurations, the explicit approach offers more flexibility and control.
# The Matplotlib community recommends using explicit programming for these
# tasks.
#
# Generalized Function Guidelines
# -------------------------------
#
# For users that have recurring plots and graphs, the signature function
# similar to the format below serves as a reusable template.


def my_plotter(ax, data1, data2, param_dict):
    """
    Helper function to make a graph.

    Parameters
    ----------
    ax : Axes
        Specific Axes to graph data to
    data1 : array
        X data
    data2 : array
        Y data
    param_dict : dict
        Dictionary of keyword arguments passes to method

    Returns
    -------
    out : list
        List of Artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

##############################################################################
#
#
# Additional Resources
# ====================
#
# - :ref:`tutorials`
#   More on detailed guides and specific topics.
# - :ref:`gallery`
#   Collection of visualizations and demonstrative examples.
# - `External Resources <https://matplotlib.org/resources/index.html>`_
#   Curated content from other users.
#
