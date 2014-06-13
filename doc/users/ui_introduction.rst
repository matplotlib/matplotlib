.. _ui_introduction:

Introduction
============

Matplotlib can be placed into a variety of user-interface frameworks (or user-interface toolkits), which allows for the rapid development of applications that visualize data. The user dynamically change the appearance of the plot and can even manipulate the data through the plot. The integration of Matplotlib into web-frameworks brings these same features to webpages. In addition, many rendering backends (or graphical libraries) are supported, which can be used for dynamic applications or exporting the graphs into static documents.

The different user-interface frameworks offer different features and programming-language bindings. Most offer graphical user-interface-builders (or rapid-application-development programs) that are useful for constructing the layout of larger applications. The appearance of the user interface will be, by default, similar to the style of the operating system, but can be customized.

The following user-interface frameworks are supported (graphical builders in brackets):

- Gtk+
  
    - Gtk+ 2.x (Glade 3.8.x)
    - Gtk+ 3.x (Glade 3.16.x)

- macosx

- Qt (Qt-Creator)
  
    - Qt4
    - Qt5

- Tk

- wxWidgets (wxFormbuilder)

    - wx 2.x
    - wx 3.x

    *The wxWidget-framework is, as of June 2014, the only toolkit that does not support Python 3.x*
  
The supported web-frameworks are:

- Django

The following backends are supported:

- Agg (Anti-Grain Geometry)

    - CocoaAgg
    - GTKAgg
    - TkAgg
    - QtAgg
    - WebAgg
    

- Cairo
- Gdk (GIMP Drawing Kit)
- Pdf (Portable Document Format)
- Pgf (Portable Graphics Format)
- Ps (PostScript)
- Svg (Scalable Vector Graphics)

.. figure:: ../_static/mpl_with_glade_3.png
    :width: 50 %
    :alt: Figure of the Glade interface, showing how to divide the window into 2 scrolled windows.
    :align: center

    **Figure 1:** Example of Matplotlib embedded as a widget in a user-interface framework. This example shows the GTK+ 3.x framework under Ubuntu 14.04. *Note that the appearance of the window will depend on the operating system and can additonialy be customized.*