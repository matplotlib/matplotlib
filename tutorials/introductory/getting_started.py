"""
***************
Getting Started
***************

This tutorial covers some basic usage patterns and best-practices to
help you get started with Matplotlib.
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
# information regarding manipulating backend capabilities, see """ref""".
#
# There are two main ways of producing graphs with Matplotlib, explicit and
# implicit. Explicit code, using Object Oriented Programmiong (OOP), and
# implicit code, using `pyplot`, are the foundation for creating and
# manipulating data into visualizations.
#
# Explicit programming, OOP, helps users generalize code and is useful for
# repeated uses or larger projects. This is also a more robust way of
# controlling customizations for visualizations. Users looking to have control
# over every part of the graph can call methods on each item.
#
# The implicit `pyplot` approach to generate plots is simple. It is helpful
# for basic plots and for interactive environments, such as Jupyter Notebooks.
# Users familiar with MATLAB or would like to have Matplotlib automatically
# create and manage parts of the visualization benefit from using `pyplot`
# functions to plot their data.
#
#


##############################################################################
#
# Requirements
# ============
#
# Matplotlib is a Python library and an installed version of Python 3.6 or
# higher is required. Depending on your operating system, Python may already
# be installed on your machine.
#
# Installing Maptlotlib is required in order to generate plots with the
# library. You can install Matplotlib for your own development environment(s)
# or use a third-party package distribution.
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
# In order to install Matplotlib from the source directory, you can run the
# following command line executions using Python and installer program `pip`
# for the latest version of Matplotlib and its dependencies. This will compile
# the library from the current directory on your machine.
#
# `python -m pip install matplotlib`
#
# .. note::
#
#     If you would like to contribute to Matplotlib, see the developer
#     installation guide for details about the process.
#
# Interactive environments
# ------------------------
#
# The Matplotlib community suggests using `IPython <https://ipython.org/>`_
# through `Jupyter <https://jupyter.org/index.html>`_ as the primary
# interactive environment.

##############################################################################
#
# Plotting
# ========
#
# The common conventions for preparing to plot data involve importing the
# necessary libraries with abbreviations for convenience. Both implicit and
# explicit programming require the following.

import matplotlib.pyplot as plt

##############################################################################
#
# The `pyplot` module in Matplotlib is a collection of functions. The module's
# functions create, manage, and manipulate the current figure and the plotting
# area.
#
# These are the two common strategies for creating plots with Matplotlib.
#
# * Explicit: Code has explicit references to objects. Users manage objects
#   for specific figures and axes and call on methods for manipulating data.
#     * Object-oriented programming (OOP), robust control and useful for
#       generalized code.
#
# * Implicit: The programming is designed to remember preceding events or
#   interactions. Matplotlib automatically manages figures and axes.
#     * `pyplot`, most similar to MATLAB and convenient for interactive
#       environments.
#
# .. note::
#
#     The Matplotlib community does not recommend interchanging explicit and
#     implicit strategies. When using one as standard, all code should follow
#     the same strategy. Switching back and forth between explicit and
#     implicit programming can yield errors.
#
# For other techniques of creating plots with Matplotlib, refer to
# """ref""".
#
# Data
# ----
#
# The Matplotlib library manages data in the form of iterables and/or
# sequenced items. These can also take the form of NumPy arrays like
# `numpy.array` or `numpy.ma.masked_array`. All plotting functions take these
# data structures.
#

# Sample Data

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
# of the `pyplot` module to unpack a set or sets of explicit figure and axes.
# Each axes has its own methods to graph data. In addition, each axes also
# uses separate methods to create and manage parts of a figure. These methods
# are different from those of the implicit programming approach.

# Explicit programming with OOP

x = months
y1 = income
y2 = chk_acct_09
y3 = svg_acct_09

fig, ax = plt.subplots()  # Figure & axes unpacked separately with module.

ax.plot(x, y1, label='Checking Account')
ax.plot(x, y2, label='Savings Account')
ax.plot(x, y3, label='Income')
ax.set_xlabel('Month')  # Axes use separate methods to manage parts of figure.
ax.set_ylabel('USD')
ax.set_title('Personal Financial Tracking from 2010')
ax.legend()

##############################################################################
#
# For the OOP example, the figure and axes are unpacked from the module using
# a single instance of `pyplot`. This convention uses `plt.subplots()` and
# defaults to one figure, `fig`, and one axes, `ax`. The section below on
# customizations contains additional information about multiple visulizations
# and other modifications.
#
# Using the OOP approach allows for `fig` and `ax` to use separate methods to
# manipulate the visualization. Instead of using the module `pyplot` for all
# instances of managing objects, the specfic axes refers to OOP usage and
# manages the respective data.
#
# Implicit: `pyplot`
# ------------------
#
# To use implicit programming for Matplotlib involves using the `pyplot`
# module. The figure and axes are automatically generated by the module.
# Pass data through as arguments using methods within the module.
# Additional parts of the figure are also available through the module
# methods.

# Implicit programming with pyplot

y4 = chk_acct_10
y5 = svg_acct_10

plt.plot(x, y1, label='Income')
plt.plot(x, y4, label='Checking Account')
plt.plot(x, y5, label='Savings Account')
plt.xlabel('Month')  # Module methods generate parts of figure.
plt.ylabel('USD')
plt.title("Personal Financial Tracking from 2009")
plt.legend()

##############################################################################
#
# In the example above, the `pyplot` module contains its own methods of
# actionable tasks for the data. The `plt.plot` plots data as a line graph
# with various keyword arguments as customizable options. The module also
# includes other methods for generating parts of the visualization. These parts
# use different methods from the OOP approach.
#
# .. note::
#
#    The names and spelling for methods may be similar for both `pyplot` and
#    OOP approaches. Errors may occur when using the wrong corresponding
#    method. Confirm with the documentation API for specific method names
#    according to your programming.

##############################################################################
#
# Customizations
# ==============
#
# There are two main parts to building a visualization with Matplotlib, the
# figure and the axes.
#
# Components of Matplotlib Figure
# -------------------------------
#
# The image below depicts each visible element of a Matplotlib graph.
#
# * Figure
#    * The figure is the working space for the programming. All visible
#      objects on a graph are located within the figure.
# * Axes
#    * Axes are subplots within the figure. They contain figure elements and
#      are responsible for plotting and configuring additional details.
#       * Note: Axes and Axis are not synonymous. Axis refers to
#         """ref""".
#
# Multiple Graphs within a Figure
# -------------------------------
#
# For multiple graphs using a single figure, explicit and implicit approaches
# use a similar convention for mapping out multiple axes. Matplotlib manages
# more than one axes in a two-dimensional matrix. They are arranged by row
# amount and then by column amount. The third argument represents the specific
# axes involved.
#
# When looking for more complex solutions to multiple graphs within a figure,
# use the GridSpec module to organize the layout.

# Explicit with OOP

fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row',
                               figsize=[8, 4], constrained_layout=True)

fig.suptitle('Personal Financial Tracking \'09 - \'10',
             fontsize=16, weight='black')

ax1.plot(x, y1, label='Income')
ax1.plot(x, y2, label='Checking')
ax1.plot(x, y3, color='green', label='Savings')
ax1.set_xticklabels(months, rotation=270)
ax1.set_title('2009', fontsize='small')
ax1.legend(loc='upper left')

ax2.plot(x, y1, label='Income')
ax2.plot(x, y2, label='Checking')
ax2.plot(x, y3, color='green', label='Savings')
ax2.set_xticklabels(months, rotation=270)
ax2.set_title('2010', fontsize='small')

##############################################################################
#
# The OOP example above also uses two axes to graph the data. However, the OOP
# approach must refer to an explicitly generated axes after creating both the
# figure and axes.

# Implicit with pyplot

plt.subplot(1, 2, 1)  # Note the different method name for implicit.
plt.plot(x, y1, label='Income')
plt.plot(x, y2, label='Checking')
plt.plot(x, y1, color='green', label='Savings')
plt.xticks(x, months, rotation=270)
plt.title('2009', fontsize='small')

plt.subplot(1, 2, 2)
plt.plot(x, y1, label='Income')
plt.plot(x, y4, label='Checking')
plt.plot(x, y5, color='green', label='Savings')
plt.xticks(x, months, rotation=270)
plt.title('2009', fontsize='small')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.suptitle('Personal Financial Tracking', weight='black')
plt.tight_layout()

##############################################################################
#
# The `pyplot` example above uses two axes to graph data. In each instance,
# Matplotlib auotmatically manages the specific axes so that each action of
# plotting data does not interfere with the previous instance.
