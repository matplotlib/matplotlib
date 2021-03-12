===========================
Matplotlib for Excel users
===========================

Introduction
--------------

Microsoft Excel is a spreadsheet program from Microsoft with the capability to make charts of the different data inside the cells the user makes. Some users may have a familiarity with the Office program and might want to change to matplotlib for data plotting. This document aims to guide new matplotlib users by referencing the graphical capabilities of Microsoft Excel.

Some key differences
---------------------

+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| Excel                                                                                 | Matplotlib                                                                                                    |
+=======================================================================================+===============================================================================================================+
| In Excel commands and scripting is written in Visual Basic for Applications (VBA).    | Matplotlib is written in Python and uses Python for all its commands.                                         |
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| In Excel you can use cells to create formulas.                                        | In matplotlib you can use different datastructures to store your data.                                        |
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| You can use Excel to calculate formulas.                                              | For matplotlib you can use the core mechanics of Python but for more complex formulas NumPy or SciPy is used. |
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
| In Excel you can use an user interface for tweaking the charts.                       | In matplotlib you need to use written code.                                                                   |
+---------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+

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


The most basic plot would look like the following in written code:

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

+-----------------------+-------------------------------+
| Excel                 | Matplotlib                    |
+=======================+===============================+
| Column                | ``plt.bar(x,y)``              |
+-----------------------+-------------------------------+
| Stacked Column        | ``plt.bar(x,y)``              |
|                       | ``plt.bar(x,z,bottom=y)``     |
+-----------------------+-------------------------------+
| Line                  | ``plt.plot(x,y)``             |
+-----------------------+-------------------------------+
| Stacked Line          | ``plt.stackplot(x,y,z)``      |
+-----------------------+-------------------------------+
| Line with markers     | ``plt.plot(x,y,'-o')``        |
+-----------------------+-------------------------------+
| Pie                   | ``plt.pie(y,labels=x)``       |
+-----------------------+-------------------------------+
| Bar                   | ``plt.barh(x,y)``             |
+-----------------------+-------------------------------+
| Stacked Bar           | ``plt.barh(x,y)``             |
|                       | ``plt.barh(x,z,left=y)``      |
+-----------------------+-------------------------------+
| Area                  | ``plt.plot(x,y)``             |
|                       | ``plt.fill_between(x,0,y)``   |
+-----------------------+-------------------------------+
| Scatter               | ``plt.scatter(x,y)``          |
+-----------------------+-------------------------------+
| Radar                 | ``plt.polar(theta,y)``        |
+-----------------------+-------------------------------+
| Radar with markers    | ``plt.polar(theta,y,'-o')``   |
+-----------------------+-------------------------------+
| Histogram             | ``plt.hist(y)``               |
+-----------------------+-------------------------------+
| Box and Whisker       | ``plt.boxplot(y)``            |
+-----------------------+-------------------------------+