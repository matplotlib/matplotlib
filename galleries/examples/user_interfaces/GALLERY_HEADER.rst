.. _user_interfaces:

Embedding Matplotlib in graphical user interfaces
=================================================

You can embed Matplotlib directly into a user interface application by
following the embedding_in_SOMEGUI.py examples here. Currently
Matplotlib supports PyQt/PySide, PyGObject, Tkinter, and wxPython.

When embedding Matplotlib in a GUI, use the Matplotlib API directly. For most
GUI integrations, this means creating standalone figure objects via
``fig = Figure()`` rather than using pyplot-managed figures such as
``plt.figure()`` or ``plt.subplots()``. The examples below show how to connect
these figures to the supported GUI toolkits.
