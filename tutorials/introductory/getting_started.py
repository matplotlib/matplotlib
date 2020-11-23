"""

.. _getting_started:

***************
Getting Started
***************

This tutorial covers basic usage patterns and best-practices to help you get
started with Matplotlib.
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
# methods. This process occurs on the backend and is not user-facing. For more
# information regarding manipulating backend capabilities, see
# :ref:`backends`.
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
# | Users explicitly create and manage | Automatically manages Figure and   |
# | all Figure elements.               | Axes.                              |
# +------------------------------------+------------------------------------+
# | Useful for repeated code use,      | Helpful for quickly graphing data  |
# | generalization, robust             | when using interactive             |
# | configurations of graphs.          | environments.                      |
# +------------------------------------+------------------------------------+
# | Recommended to new users for       | Most useful for users coming from  |
# | learning fundamentals.             | MATLAB.                            |
# +------------------------------------+------------------------------------+
#
# Explicit programming helps users generalize code and is useful for repeated
# uses or larger projects. This is also a more robust way of controlling
# customizations for visualizations. Users looking to have control over every
# part of the graph can call methods on each item. Most users benefit using
# explicit programming for regular Matplotlib use as the user manages each
# element of building a graph.
#
# Implicit programming with ``pyplot`` is simpler. It is helpful for basic
# plots and for interactive environments, such as Jupyter Notebooks. Users
# familiar with MATLAB or would like to have Matplotlib automatically create
# and manage parts of the visualization benefit from using the ``pyplot``
# module to graph data. New users to Matplotlib may experience difficulty
# understanding how elements of a visualization work together when using the
# implicit approach.
#
# Examples
# --------
#
# +------------------------------------+------------------------------------+
# | Explicit, Object Oriented          | Implicit, ``pyplot``               |
# | Programming (OOP)                  |                                    |
# +====================================+====================================+
# | ::                                 | ::                                 |
# |                                    |                                    |
# |     fig, ax = plt.subplots()       |    plt.plot([1,2,3],[1,2,3])       |
# |     ax.plot([1,2,3],[1,2,3])       |                                    |
# |                                    |                                    |
# +------------------------------------+------------------------------------+
# | `.pyplot.subplots` generates a     | :mod:`matplotlib.pyplot` creates   |
# | `~.figure.Figure` and one or       | implicit Figure and Axes if        |
# | more `~.axes.Axes` explicitly.     | there are no pre-existing          |
# | `.Axes.plot` plots the data.       | elements and `.pyplot.plot` plots  |
# |                                    | the data.                          |
# +------------------------------------+------------------------------------+
#
#
# Requirements
# ============
#
# Matplotlib is a Python library and an installed version of *Python 3.6 or
# higher* is required. Depending on your operating system, Python may already
# be installed on your machine.
#
# Installing Maptlotlib is required in order to generate graphs with the
# library. Install Matplotlib for your own development environment manually or
# use a third-party package distribution.
#
# Third-party package distributions, such as
# `Anaconda <https://www.anaconda.com/>`_,
# `ActiveState <https://www.activestate.com/activepython/downloads>`_,
# or `WinPython <https://winpython.github.io/>`_,
# already provide Matplotlib and its dependencies. They have the added benefit
# of including other scientific Python libraries as well. These packages work
# as is and do not require additional installations.
#
# Installation from source
# ------------------------
#
# In order to install Matplotlib from the source directory, run the
# following command line executions using Python and installer program ``pip``
# for the latest version of Matplotlib and its dependencies. This will compile
# the library from the current directory on your machine. Depending on your
# operating system, you may need additional support.
#
# ``python -m pip install matplotlib``
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
# convenience. Both explicit and implicit programming require the following
# code.

import matplotlib.pyplot as plt
import numpy as np

##############################################################################
#
# The ``pyplot`` module in Matplotlib is a collection of functions. The
# module's functions create, manage, and manipulate the current Figure and the
# plotting area.
#
# NumPy is a common scientific Python library that benefits users with
# additional robust tools for manipulating data.
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
# | * Code has explicit references to  | * The programming is designed to   |
# |   objects. Users manage objects for|   remember preceding events or     |
# |   the specific Figure and Axes and |   interactions. This preserved     |
# |   call on methods for manipulating |   state allows Matplotlib to       |
# |   data.                            |   automatically manage a Figure and|
# | * Object Oriented Programming      |   Axes.                            |
# |   allows for robust control and is | * The module ``pyplot`` operates   |
# |   useful for generalized code.     |   similarly to MATLAB and is       |
# |                                    |   convenient for interactive       |
# |                                    |   environments.                    |
# +------------------------------------+------------------------------------+
#
# .. note::
#
#     The Matplotlib community does not recommend interchanging explicit and
#     implicit strategies. When using one as standard, all code should follow
#     the same strategy. Switching back and forth between explicit and
#     implicit programming can yield errors.
#
# For other techniques of creating plots with Matplotlib, refer to
# :ref:`user_interfaces`.
#
# Data
# ----
#
# The Matplotlib library manages data in the form of iterables and/or
# sequenced items. These can also take the form of NumPy arrays like
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
#    Other containers, such as `pandas` data objects, may not work as
#    intended.
#
# Explicit: Object Oriented Programming (OOP)
# --------------------------------------------
#
# To use explicit programming for Matplotlib involves using a single instance
# of the ``pyplot`` module. This unpacks a set of an explicit Figure and Axes.
# There can be more than one Axes; however, each Axes can only be on one
# Figure.
#
# Each Axes has its own methods to graph data. In addition, each Axes
# also uses separate methods to create and manage parts of a Figure. These
# methods are different from those of the implicit programming approach.

# Explicit programming with OOP

x = months
y1 = income
y2 = chk_acct_09
y3 = svg_acct_09
# Assigning sample data to labeled variables.

fig, ax = plt.subplots()
# Explicit Figure & Axes unpacked separately with module.
# Conventional object abbreviations are `fig` and `ax`, respectively.

ax.plot(x, y1, label='Checking Account')
ax.plot(x, y2, label='Savings Account')
ax.plot(x, y3, label='Income')
# Single explicit Axes graphs multiple data points.

ax.set_xlabel('Month')
ax.set_ylabel('USD')
ax.set_title('Personal Financial Tracking from 2009')
ax.legend()
# Explicit Axes use separate methods to manage parts of Figure.

plt.show()
# The pyplot module displays the Figure.

##############################################################################
#
# For the OOP example, the Figure and Axes are unpacked from the module using
# a single instance of ``pyplot``. This convention uses ``plt.subplots()``
# and defaults to one Figure, ``fig``, and one Axes, ``ax``. The
# `Customizations`_ section below contains additional information about
# multiple visulizations and other modifications.
#
# Using the OOP approach allows for ``fig`` and ``ax`` to use separate methods
# to manipulate the visualization. Instead of using the module ``pyplot`` for
# all instances of managing objects, the specfic Axes refers to OOP usage and
# manages the respective data.
#
# Implicit: ``pyplot``
# --------------------
#
# Implicit programming for Matplotlib centers around using the ``pyplot``
# module. The Figure and Axes are automatically generated by the module.
# The methods and functions within the module take incoming data as arguments.
# Additional parts of the Figure are also available through the module
# methods.

# Implicit programming with pyplot

y4 = chk_acct_10
y5 = svg_acct_10
# Assigning former data to labeled variable.
# Previous variables are still referenced.

plt.plot(x, y1, label='Income')
plt.plot(x, y4, label='Checking Account')
plt.plot(x, y5, label='Savings Account')
# Module plots multiple data points on implicitly generated Axes.

plt.xlabel('Month')
plt.ylabel('USD')
plt.title("Personal Financial Tracking from 2010")
plt.legend()
# Module methods generate parts of Figure.

plt.show()
# The module displays the Figure.

##############################################################################
#
# In the example above, the ``pyplot`` module contains its own methods of
# actionable tasks for the data. The ``plt.plot`` plots data as a line graph
# with various keyword arguments as customizable options. The module also
# includes other methods for generating parts of the visualization. These parts
# use different methods from the OOP approach.
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
# visualization. The source code is available as
# :ref:`sphx_glr_gallery_showcase_anatomy.py`.
#
# .. note::
#
#     ``Figure`` and ``Axes`` identify empty regions of the diagram;
#     however, these elements are foundational in operation.
#
# .. image:: ../../_static/anatomy.png
#
# :class:`~matplotlib.figure.Figure`
#
# The Figure is the working space for the programming. All visible
# objects on a graph are located within the Figure.
#
# :class:`~matplotlib.axes.Axes`
#
# Axes are subplots within the Figure. They contain Figure elements and
# are responsible for plotting and configuring additional details. Each
# Figure can contain multiple Axes, but each Axes is specific to one
# Figure.
#
# In a Figure, each Axes can contain any number of visual elements. An Axes may
# have more than one type of visualization of data. From the `Plotting`_
# section above, the Axes in both explicit and implicit strategies contain
# multiple types of visualizations of data on a single Axes. Each of these
# types are specific to the Axes they are in.
#
# Other Components
# ^^^^^^^^^^^^^^^^
#
# :class:`~matplotlib.artist.Artist`
#
# Artists are broad Matplotlib objects that display visuals. These are the
# visible elements when the Figure is rendered. They correspond to a specific
# Axes and cannot be shared or transferred. In Matplotlib programming, all
# objects for display are Artists.
#
# .. note::
#
#   Axes and Axis are not synonymous. Axis refers to
#   :class:`~matplotlib.axis.Axis`.
#
# Manipulating Artists
# --------------------
#
# With simple plots, Matplotlib automatically generates the basic elements of
# a graph. Artists as objects allow for more control over the visual.
#
# Matplotlib generates additional visual elements as Artists in the form of
# objects. As Artists, each has its own respective methods and functions.
# Explicit and implicit approaches use different methods and are not
# interchangeable.
#
# The table below compares common Artists and their different methods.
#
# +-----------------------+--------------------------+------------------------+
# | Artist                | Explicit                 | Implicit               |
# +=======================+==========================+========================+
# | Visible elements from | Each specific Axes has   | The ``pyplot`` module  |
# | rendered Figure.      | its own method for       | manages Artists based  |
# |                       | artists.                 | on most recent Figure  |
# |                       |                          | or Axes.               |
# +-----------------------+--------------------------+------------------------+
# | X-axis labels         | ``ax.set_xticks()``      | ``plt.xlabel()``       |
# |                       | ``ax.set_xticklabels()`` |                        |
# +-----------------------+--------------------------+------------------------+
# | Y-axis labels         | ``x.set_yticks()``       | ``plt.ylabel()`        |
# |                       | ``ax.set_yticklabels()`` |                        |
# +-----------------------+--------------------------+------------------------+
# | Title (Axes)          | ``ax.set_title()``       | ``plt.title()``        |
# +-----------------------+--------------------------+------------------------+
# | Legend (Axes)         | ``ax.legend()``          | ``plt.legend()``       |
# +-----------------------+--------------------------+------------------------+
#
# .. note::
#
#     In explicit programming, ``ax`` refers to an assigned variable for a
#     specific Axes. Also, axis labels require separate setting actions for
#     each specific Axes.
#
#     In implicit programming, the ``pyplot`` module automatically manages
#     separate setting actions for state-based Matplotlib objects.
#
#
# Pie Chart Examples
# ------------------
#
# Matplotlib pie charts create wedges based on data and manipulate the size of
# the Artists based on the ratio of the slice to the sum of the data. The
# ``.pie()`` method is similar for both explicit and implicit approaches.
#
# The code below illustrates various levels of configuration in keyword
# arguments as well as Artist methods for both explicit and implicit
# programming.
#
# See `matplotlib.axes.Axes.pie` and `matplotlib.pyplot.pie` for more
# information about the APIs for explicit and implicit, respectively.

# Data

budget = [475, 300, 125, 50]
# Data points are represented in wedge size compared to total sum.

descriptions = ['Shared house in Philadelphia',
                'Dog costs, phone, utilities',
                'Groceries & takeout',
                'Treasury bonds']
categories = ['Rent', 'Bills', 'Food', 'Savings']
# These lists of strings contribute to labeling corresponding to data.

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
# Hex color codes determine respective wedge color.

explode = [0, 0.1, 0.15, 0.35]
# Float list represents percentage of radius to separate from center.


def autopct_format(percent, group):
    """
    Takes percent equivalent and calculates original value from data.
    Returns fstring of value above percentage.
    """
    value = int(percent/100.*np.sum(group))
    return f'${value:<4}\n{percent:1.1f}%'
# This function is used as a lambda for formatting labels in wedges.

##############################################################################
#
# The following two plots are identical. Both the explicit and implicit
# approaches generate the exact same plot when using the same variables.
#
# Basic
# ^^^^^

# Explicit


fig, ax = plt.subplots()

ax.pie(budget, colors=colors, labels=categories)
ax.legend()
ax.set_title('Average Monthly Income Expenses')
ax.axis('equal')

plt.show()

##############################################################################
#
#

plt.pie(budget, colors=colors, labels=categories)
plt.legend()
plt.axis('equal')
plt.title('Average Monthly Income Expenses')
plt.show()

##############################################################################
#
# .. note::
#
#   There are minor differences in the method names. Overall, each method
#   performs the same action through the different approaches.
#
# Additional Configurations
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Many methods contain optional keyword arguments for further configuration.
# In the explicit example below, there are values and functions in keyword
# arguments that format the Artists.

fig, ax = plt.subplots()

ax.pie(budget,
       colors=colors,
       explode=explode,
       labels=categories,
       autopct=lambda pct: autopct_format(pct, budget),
       startangle=-80,
       shadow=True)


ax.legend()
ax.axis('equal')
ax.set_title('Average Monthly Income Expenses')
plt.show()

##############################################################################
#
#

fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(budget, labels=descriptions,
                                  colors=colors, explode=explode,
                                  autopct='%d', startangle=45,
                                  pctdistance=0.85, labeldistance=1.125,
                                  wedgeprops=dict(width=0.3),
                                  shadow=True)

for text, value in zip(autotexts, budget):
    text.set_text(f'${value}\n{text.get_text()}%')
    text.set_fontweight('medium')

ax.legend(wedges, categories, title='Categories',
          bbox_to_anchor=(0.125, 0.5), loc='center right')
ax.set_title('Average Monthly Income Expenses')
ax.axis('equal')

plt.show()

##############################################################################
#
#
# Customization
# =============
#
# Multiple Graphs within a Figure
# -------------------------------
#
# For multiple graphs using a single Figure, explicit and implicit approaches
# use a similar convention for mapping out multiple Axes. Matplotlib manages
# more than one Axes in a two-dimensional matrix. They are arranged by row
# amount and then by column amount. The third argument represents the specific
# Axes involved.
#
# When looking for more complex solutions to multiple graphs within a Figure,
# use the :class:`matplotlib.gridspec.GridSpec` module to organize the layout.
#
# Explicit
# ^^^^^^^^

# Explicit with OOP

fig, (ax1, ax2) = plt.subplots(1, 2,
                               sharey='row',
                               figsize=[8, 4],
                               constrained_layout=True)
# Figure and two Axes unpacked from arguments (1, 2) as row & column.
# Keyword arguments provide additional details of sharing Y-axis, Figure size
# and layout formatting.

fig.suptitle('Personal Financial Tracking \'09 - \'10')
# Explicit Figure object has separate method for title.

ax1.plot(x, y1, label='Income')
ax1.plot(x, y2, label='Checking')
ax1.plot(x, y3, color='green', label='Savings')
# First explicit Axes object plots data with additional keyword arguments.

ax1.set_xticks(months)
ax1.set_xticklabels(months, rotation=270)
ax1.set_title('2009', fontsize='small')
ax1.legend(loc='upper left')
# First explicit Axes object uses separate methods for ticks on the X-axis,
# title, and legend. Keyword arguments are for additional configurations.

ax2.plot(x, y1, label='Income')
ax2.plot(x, y4, label='Checking')
ax2.plot(x, y5, color='green', label='Savings')
# Explicit second Axes object plots data similarly to first explicit Axes.

ax2.set_xticks(months)
ax2.set_xticklabels(months, rotation=270)
ax2.set_title('2010', fontsize='small')
# Explicit second Axes object has separate methods as well.

plt.show()
# The pyplot module displays the Figure.

##############################################################################
#
# The OOP example above also uses two Axes to graph the data. However, the OOP
# approach must refer to an explicitly generated Axes after creating both the
# Figure and Axes.
#
# In the unpacking process, numerous Axes can also be assigned to the single
# variable. To reference a specific Axes, you can index the location of the
# respective Axes as a matrix through the single variable.
#
# .. note::
#
#   The code below demonstrates indexing multiple Axes.
#   ::
#
#       fig, ax = plt.subplots(2,2)
#
#       ax[0,0].plot([1,2,3],[1,2,3])
#       ax[0,1].plot([3,2,1],[3,2,1])
#       ax[1,0].plot([3,2,1],[3,2,1])
#       ax[1,1].plot([1,2,3],[1,2,3])
#
# Implicit
# ^^^^^^^^

# Implicit with pyplot

plt.subplot(1, 2, 1)
# Module implicitly manages a matrix size of (1, 2) for row & column
# to work on the first implicit Axes.

plt.plot(x, y1, label='Income')
plt.plot(x, y2, label='Checking')
plt.plot(x, y3, color='green', label='Savings')
# Module plots data on first implicit Axes.

plt.xticks(x, months, rotation=270)
plt.title('2009', fontsize='small')
# Module methods generate parts of Figure for first implicit Axes.

plt.subplot(1, 2, 2)
# Module implicitly manages matching matrix size to work on second implicit
# Axes.

plt.plot(x, y1, label='Income')
plt.plot(x, y4, label='Checking')
plt.plot(x, y5, color='green', label='Savings')
# Module plots data on second implicit Axes.

plt.xticks(x, months, rotation=270)
plt.title('2009', fontsize='small')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# Module methods generate parts of Figure for second implicit Axes.

plt.suptitle('Personal Financial Tracking')
plt.tight_layout()
# Module methods for managing Figure elements.

plt.show()
# Module displays the Figure.

##############################################################################
#
# The ``pyplot`` example above uses two Axes to graph data. In each instance,
# Matplotlib auotmatically manages the specific Axes so that each action of
# plotting data does not interfere with the previous instance.
#
# .. note::
#
#   There are limitations for customizing the implicit approach without
#   referencing specific Axes and Artists within the Figure. For more
#   advanced configurations, the explicit approach is recommended.
#
# Generalized Function Guidelines
# -------------------------------
#
# For users with that have recurring plots and graphs, the Matplotlib
# community recommends a signature function similar to the format below.


def my_plotter(ax, data1, data2, param_dict):
    """
    Parameters
    ----------
    :param ax: Axes
    :param data1: array of X data
    :param data2: array of Y data
    :param param_dict: Dictionary of keyword arguments to pass to method

    Returns
    -------
    :returns: out : list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    # Note: Other methods from Axes class are also applicable.
    return out

##############################################################################
#
# .. currentmodule:: getting_started
# .. autofunction:: my_plotter
#
# Additional Resources
# ====================
#
