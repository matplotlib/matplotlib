.. _user_interfaces:

Embedding Matplotlib in graphical user interfaces
=================================================

You can embed Matplotlib directly into a user interface application by
following the embedding_in_SOMEGUI.py examples here. Currently
Matplotlib supports PyQt/PySide, PyGObject, Tkinter, and wxPython.

When embedding Matplotlib in a GUI, use standalone `.Figure` objects only.
Do not use pyplot, because you do not want to depend on its globally managed
state and its event loop control. In particular, use ``fig = Figure()`` rather
than ``plt.figure()``, and ``fig = Figure(); axs = fig.subplots()`` rather than
``fig, axs = plt.subplots()``. The examples below show how to connect these
figures to the supported GUI toolkits.
