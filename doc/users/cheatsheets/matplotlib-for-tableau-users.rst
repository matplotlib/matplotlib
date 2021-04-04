.. role:: raw-html(raw)
    :format: html

=============================
Matplotlib for Tableau users
=============================

.. contents::
    :local:

Introduction
--------------

Tableau is a Business Intelligence tool that provides a highly customizable dashboard to preview your data, being highly customizable. 
Although the use case is slightly different, the visualization principles stay the same, with the same idioms and best practice, but a
different environment and API.

Basics
--------

When coming from a GUI heavy software like Tableau a programming language might be intimidating. In Tableau, all your data is stored in your datasets. In Python and therefore also matplotlib your data is stored inside arrays and lists. When you have experience with Tableau you might also like to work with dictionaries (tableau_dictionary_example_). But let's first look into arrays. To write an array by hand with values from 1 to 10 will look in Python like this\:

::

    x = [1,2,3,4,5,6,7,10]

Easy right? To call a perticular value of x you need to call x with the correct index. This will look something like this\:

::

    print(x[0]) # This will print 1
    print(x[1]) # This will print 2
    print(x[2]) # This will print 3
    print(x[3]) # This will print 4
    # etc...

When calling these values you can also use these values for different calculations. I suggest now looking into the `Python Beginners Guide <https://wiki.python.org/moin/BeginnersGuide>`_ if you are not yet familiar with Python itself. If you know your way with Python but want to do some calculations (like sum) with your data before plotting I suggest looking into the `Numpy Beginners Guide <https://numpy.org/doc/stable/user/absolute_beginners.html>`_.

In the section plots_and_charts_tableau_ you will find a list of references to the different plots that matplotlib has to offer that look like Tableau graphs. Feel free to use them any time you want. For more information on the usage of matplotlib see :doc:`/tutorials/introductory/usage`.

.. _plots_and_charts_tableau:


Plots and charts
------------------

+-------------------------------+-----------------------------------------------------------------------+
| Tableau                       | Matplotlib                                                            |
+===============================+=======================================================================+
| Simple Bar Chart              | `~.Axes.bar`\(x,y)                                                    |
+-------------------------------+-----------------------------------------------------------------------+
| Stacked Column                | `~.Axes.bar`\(x,y) :raw-html:`<br />` `~.Axes.bar`\(x,z,bottom=y)     |
+-------------------------------+-----------------------------------------------------------------------+
| Simple Line Chart             | `~.Axes.plot`\(x,y)                                                   |
+-------------------------------+-----------------------------------------------------------------------+
| Multiple Measure Line Chart   | `~.Axes.stackplot`\(x,y,z)                                            |       
+-------------------------------+-----------------------------------------------------------------------+
| Line with markers             | `~.Axes.plot`\(x,y,'-o')                                              |
+-------------------------------+-----------------------------------------------------------------------+
| Simple Pie Chart              | `~.Axes.pie`\(y,labels=x)                                             |
+-------------------------------+-----------------------------------------------------------------------+
| Bar                           | `~.Axes.barh`\(x,y)                                                   |
+-------------------------------+-----------------------------------------------------------------------+
|Stacked Bar Chart              | `~.Axes.barh`\(x,y) :raw-html:`<br />` `~.Axes.barh`\(x,z,left=y)     |
+-------------------------------+-----------------------------------------------------------------------+
| Area                          | `~.Axes.plot`\(x,y) :raw-html:`<br />` `~.Axes.fill_between`\(x,0,y)  |
+-------------------------------+-----------------------------------------------------------------------+
| Simple Scatter Plot           | `~.Axes.scatter`\(x,y)                                                |
+-------------------------------+-----------------------------------------------------------------------+
| Radar                         | `~.pyplot.polar`\(theta,y)                                            |
+-------------------------------+-----------------------------------------------------------------------+
| Radar with markers            | `~.pyplot.polar`\(theta,y,'-o')                                       |
+-------------------------------+-----------------------------------------------------------------------+
| Histogram                     | `~.Axes.hist`\(y)                                                     |
+-------------------------------+-----------------------------------------------------------------------+
| Box Plot                      | `~.Axes.boxplot`\(y)                                                  |
+-------------------------------+-----------------------------------------------------------------------+
| Tree map                      | no direct correlation                                                 |
+-------------------------------+-----------------------------------------------------------------------+

.. _tableau_dictionary_example:

Introduction to dictionaries
-------------------------------

Because of your experience with Tableau, you probably want to work with big sets of data and simply an array would not suffice. The easiest way to accomplish this is with dictionaries. Let's say that you normally have a dataset that has the following format\:

+------------------+---------------------+--------------------------+---------------+
| Dog              | Number of females   | Number of males          | Total         |
+==================+=====================+==========================+===============+
| Labrador         | 5                   | 10                       | 15            |
+------------------+---------------------+--------------------------+---------------+
| Poodle           | 12                  | 7                        | 19            |    
+------------------+---------------------+--------------------------+---------------+
| Chihuahua        | 4                   | 5                        | 9             |
+------------------+---------------------+--------------------------+---------------+

This would look in a dictionary format like this\:

::

    dogs = {
        'Dog' : (Labrador,Poodle,Chihauhau),
        'Number of females' : (5,12,4),
        'Number of males' : (10,7,5),
        'Total' : (15,19,9)
    }

Then if you would like to only plot the number of females it would look like this:

::

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.bar(dogs['Dog'], dogs['Number of females']) # Plot some data on the axes.

For more information on dictionaries, you can look at the Python documentation for `dictionaries <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_.