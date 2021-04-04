===========================
Matplotlib for Excel users
===========================

.. contents::
    :local:

Introduction
--------------

Microsoft Excel is a spreadsheet program from Microsoft with the capability to make charts of the different data inside the cells the user makes. Some users may have a familiarity with the Office program and might want to change to matplotlib for data plotting. This document aims to guide new matplotlib users by referencing the graphical capabilities of Microsoft Excel.

Some key differences
---------------------

+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Excel                                                                                 | Matplotlib                                                                                                                     |
+=======================================================================================+================================================================================================================================+
| Commands and scripts are written in Visual Basic for Applications (VBA).              | Written in Python and uses Python for all its commands.                                                                        |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Cells can be used to create formulas.                                                 | Ability to use different datastructures to store your data.                                                                    |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Excel is used to calculate formulas.                                                  | Ability to use the core mechanics of Python for simple formulas, but for more complex formulas NumPy or SciPy can be used.     |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+
| Ability to use the user interface for tweaking the charts.                            | Mandatory to use written code.                                                                                                 |
+---------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------+

Basics
--------

When coming from an Office program a programming language might be intimidating. The first thing that you need to do is say goodbye to cells and say hello to arrays and lists. To get you started we will show you a basic example of an array. Let's say that you have ten values from A1\:A10. To write them in python it should look something like this\:

::
    x = [1,2,3,4,5,6,7,10]

Easy right? But in Excel you want to use your data also in other cells. To call a perticular value of x you need to call x with the correct index. This will look something like this\:

::
    print(x[0]) # This will print 1
    print(x[1]) # This will print 2
    print(x[2]) # This will print 3
    print(x[3]) # This will print 4
    # etc...

When calling these values you can also use these values for different calculations. I suggest now looking in to the `Python Beginners Guide <https://wiki.python.org/moin/BeginnersGuide>`_ if you are not yet familiar with Python itself. If you know your way with Python but want to do some calculations (like sum) with you data before plotting I suggest looking in to the `Numpy Beginners Guide <https://numpy.org/doc/stable/user/absolute_beginners.html>`_.

In the section plots_and_charts_ you will find a list of references to the different plots that matplotlib has to offer that look like Excel graphs. Feel free to use them any time you want. For more information on the usage of matplotlib see :doc:`/tutorials/introductory/usage`.

.. _plots_and_charts:

Plots and charts
------------------

+-----------------------+-------------------------------+
| Excel                 | Matplotlib                    |
+=======================+===============================+
| Column                | `~.Axes.bar`(x,y)             |
+-----------------------+-------------------------------+
| Stacked Column        | `~.Axes.bar`(x,y)             |
|                       | `~.Axes.bar`(x,z,bottom=y)    |
+-----------------------+-------------------------------+
| Line                  | `~.Axes.plot`(x,y)            |
+-----------------------+-------------------------------+
| Stacked Line          | `~.Axes.stackplot`(x,y,z)     |
+-----------------------+-------------------------------+
| Line with markers     | `~.Axes.plot`(x,y,'-o')       |
+-----------------------+-------------------------------+
| Pie                   | `~.Axes.pie`(y,labels=x)      |
+-----------------------+-------------------------------+
| Bar                   | `~.Axes.barh`(x,y)            |
+-----------------------+-------------------------------+
| Stacked Bar           | `~.Axes.barh`(x,y)            |
|                       | `~.Axes.barh`(x,z,left=y)     |
+-----------------------+-------------------------------+
| Area                  | `~.Axes.plot`(x,y)            |
|                       | `~.Axes.fill_between`(x,0,y)  |
+-----------------------+-------------------------------+
| Scatter               | `~.Axes.scatter`(x,y)         |
+-----------------------+-------------------------------+
| Radar                 | `~.Axes.polar`(theta,y)       |
+-----------------------+-------------------------------+
| Radar with markers    | `~.Axes.polar`(theta,y,'-o')  |
+-----------------------+-------------------------------+
| Histogram             | `~.Axes.hist`(y)              |
+-----------------------+-------------------------------+
| Box and Whisker       | `~.Axes.boxplot`(y)           |
+-----------------------+-------------------------------+