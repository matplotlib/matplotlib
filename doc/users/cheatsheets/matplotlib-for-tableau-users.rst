.. role:: raw-html(raw)
    :format: html

=============================
Matplotlib for Tableau users
=============================

.. contents::
    :local:

Introduction
--------------

Tableau is Business Intelligence tool that provides a highly customizable dashboard to preview your data, being highly customizable. 
Although the usecase is slightly different, the visualization principles stay the same, with the same idioms and best practice, but a
different environment and API.

Some key differences
---------------------

Basics
--------

When using matplotlib you need to import the package into your program. For this tutorial we import the pyplot package from matplotlib. It is also recommended to use NumPy for your calculations.

::

    import numpy as np
    from matplotlib import pyplot as plt

Now we can use the different functions provided by numpy and pyplot by calling np and plt respectively.

For the following plots we will use the following data stored in an array as upposed to the cells in an Excel sheet.

::

    x = [1,2,3,4,5,6,7,8,9,10] # In matplotlib you need to also define the x-axis
    y = [10,1,9,2,8,3,7,4,6,5] # Mockup data for visualising the different plots
    z = [1,10,2,9,3,8,4,7,5,6] # An extra dataset for particular plots

    theta = np.linspace(0,2*np.pi,10) # Array with length 10 and a range from 0 to 2*pi


The most basic plot would look like the following in written code\:

::

    plt.figure()
    plt.plot(x,y)
    plt.show()

For most plots you can add subsequential data by defining a new x- and y-axis or plotting again.

::

    plt.figure()
    plt.plot(x,y,x,z)
    plt.show

    # OR

    plt.figure()
    plt.plot(x,y)
    plt.plot(x,z)
    plt.show()


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
