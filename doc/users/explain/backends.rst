.. _backends:

========
Backends
========

.. _what-is-a-backend:

What is a backend?
------------------

A lot of documentation on the website and in the mailing lists refers
to the "backend" and many new users are confused by this term.
Matplotlib targets many different use cases and output formats.  Some
people use Matplotlib interactively from the Python shell and have
plotting windows pop up when they type commands.  Some people run
`Jupyter <https://jupyter.org>`_ notebooks and draw inline plots for
quick data analysis. Others embed Matplotlib into graphical user
interfaces like PyQt or PyGObject to build rich applications.  Some
people use Matplotlib in batch scripts to generate postscript images
from numerical simulations, and still others run web application
servers to dynamically serve up graphs.

To support all of these use cases, Matplotlib can target different
outputs, and each of these capabilities is called a backend; the
"frontend" is the user facing code, i.e., the plotting code, whereas the
"backend" does all the hard work behind-the-scenes to make the figure.
There are two types of backends: user interface backends (for use in
PyQt/PySide, PyGObject, Tkinter, wxPython, or macOS/Cocoa); also referred to
as "interactive backends") and hardcopy backends to make image files
(PNG, SVG, PDF, PS; also referred to as "non-interactive backends").

Selecting a backend
-------------------

There are three ways to configure your backend:

- The :rc:`backend` parameter in your :file:`matplotlibrc` file
- The :envvar:`MPLBACKEND` environment variable
- The function :func:`matplotlib.use`

Below is a more detailed description.

If there is more than one configuration present, the last one from the
list takes precedence; e.g. calling :func:`matplotlib.use()` will override
the setting in your :file:`matplotlibrc`.

Without a backend explicitly set, Matplotlib automatically detects a usable
backend based on what is available on your system and on whether a GUI event
loop is already running.  The first usable backend in the following list is
selected: MacOSX, QtAgg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg, Agg.  The last, Agg,
is a non-interactive backend that can only write to files.  It is used on
Linux, if Matplotlib cannot connect to either an X display or a Wayland
display.

Here is a detailed description of the configuration methods:

#. Setting :rc:`backend` in your :file:`matplotlibrc` file::

       backend : qtagg   # use pyqt with antigrain (agg) rendering

   See also :doc:`/tutorials/introductory/customizing`.

#. Setting the :envvar:`MPLBACKEND` environment variable:

   You can set the environment variable either for your current shell or for
   a single script.

   On Unix::

        > export MPLBACKEND=qtagg
        > python simple_plot.py

        > MPLBACKEND=qtagg python simple_plot.py

   On Windows, only the former is possible::

        > set MPLBACKEND=qtagg
        > python simple_plot.py

   Setting this environment variable will override the ``backend`` parameter
   in *any* :file:`matplotlibrc`, even if there is a :file:`matplotlibrc` in
   your current working directory. Therefore, setting :envvar:`MPLBACKEND`
   globally, e.g. in your :file:`.bashrc` or :file:`.profile`, is discouraged
   as it might lead to counter-intuitive behavior.

#. If your script depends on a specific backend you can use the function
   :func:`matplotlib.use`::

      import matplotlib
      matplotlib.use('qtagg')

   This should be done before any figure is created, otherwise Matplotlib may
   fail to switch the backend and raise an ImportError.

   Using `~matplotlib.use` will require changes in your code if users want to
   use a different backend.  Therefore, you should avoid explicitly calling
   `~matplotlib.use` unless absolutely necessary.

.. _the-builtin-backends:

The builtin backends
--------------------

By default, Matplotlib should automatically select a default backend which
allows both interactive work and plotting from scripts, with output to the
screen and/or to a file, so at least initially, you will not need to worry
about the backend.  The most common exception is if your Python distribution
comes without :mod:`tkinter` and you have no other GUI toolkit installed.
This happens on certain Linux distributions, where you need to install a
Linux package named ``python-tk`` (or similar).

If, however, you want to write graphical user interfaces, or a web
application server
(:doc:`/gallery/user_interfaces/web_application_server_sgskip`), or need a
better understanding of what is going on, read on. To make things easily
more customizable for graphical user interfaces, Matplotlib separates
the concept of the renderer (the thing that actually does the drawing)
from the canvas (the place where the drawing goes).  The canonical
renderer for user interfaces is ``Agg`` which uses the `Anti-Grain
Geometry`_ C++ library to make a raster (pixel) image of the figure; it
is used by the ``QtAgg``, ``GTK4Agg``, ``GTK3Agg``, ``wxAgg``, ``TkAgg``, and
``macosx`` backends.  An alternative renderer is based on the Cairo library,
used by ``QtCairo``, etc.

For the rendering engines, users can also distinguish between `vector
<https://en.wikipedia.org/wiki/Vector_graphics>`_ or `raster
<https://en.wikipedia.org/wiki/Raster_graphics>`_ renderers.  Vector
graphics languages issue drawing commands like "draw a line from this
point to this point" and hence are scale free. Raster backends
generate a pixel representation of the line whose accuracy depends on a
DPI setting.

Here is a summary of the Matplotlib renderers (there is an eponymous
backend for each; these are *non-interactive backends*, capable of
writing to a file):

========  =========  =======================================================
Renderer  Filetypes  Description
========  =========  =======================================================
AGG       png        raster_ graphics -- high quality images using the
                     `Anti-Grain Geometry`_ engine.
PDF       pdf        vector_ graphics -- `Portable Document Format`_ output.
PS        ps, eps    vector_ graphics -- PostScript_ output.
SVG       svg        vector_ graphics -- `Scalable Vector Graphics`_ output.
PGF       pgf, pdf   vector_ graphics -- using the pgf_ package.
Cairo     png, ps,   raster_ or vector_ graphics -- using the Cairo_ library
          pdf, svg   (requires pycairo_ or cairocffi_).
========  =========  =======================================================

To save plots using the non-interactive backends, use the
``matplotlib.pyplot.savefig('filename')`` method.

These are the user interfaces and renderer combinations supported;
these are *interactive backends*, capable of displaying to the screen
and using appropriate renderers from the table above to write to
a file:

========= ================================================================
Backend   Description
========= ================================================================
QtAgg     Agg rendering in a Qt_ canvas (requires PyQt_ or `Qt for Python`_,
          a.k.a. PySide).  This backend can be activated in IPython with
          ``%matplotlib qt``.
ipympl    Agg rendering embedded in a Jupyter widget (requires ipympl_).
          This backend can be enabled in a Jupyter notebook with
          ``%matplotlib ipympl``.
GTK3Agg   Agg rendering to a GTK_ 3.x canvas (requires PyGObject_ and
          pycairo_).  This backend can be activated in IPython with
          ``%matplotlib gtk3``.
GTK4Agg   Agg rendering to a GTK_ 4.x canvas (requires PyGObject_ and
          pycairo_).  This backend can be activated in IPython with
          ``%matplotlib gtk4``.
macosx    Agg rendering into a Cocoa canvas in OSX.  This backend can be
          activated in IPython with ``%matplotlib osx``.
TkAgg     Agg rendering to a Tk_ canvas (requires TkInter_). This
          backend can be activated in IPython with ``%matplotlib tk``.
nbAgg     Embed an interactive figure in a Jupyter classic notebook.  This
          backend can be enabled in Jupyter notebooks via
          ``%matplotlib notebook``.
WebAgg    On ``show()`` will start a tornado server with an interactive
          figure.
GTK3Cairo Cairo rendering to a GTK_ 3.x canvas (requires PyGObject_ and
          pycairo_).
GTK4Cairo Cairo rendering to a GTK_ 4.x canvas (requires PyGObject_ and
          pycairo_).
wxAgg     Agg rendering to a wxWidgets_ canvas (requires wxPython_ 4).
          This backend can be activated in IPython with ``%matplotlib wx``.
========= ================================================================

.. note::
   The names of builtin backends case-insensitive; e.g., 'QtAgg' and
   'qtagg' are equivalent.

.. _`Anti-Grain Geometry`: http://agg.sourceforge.net/antigrain.com/
.. _`Portable Document Format`: https://en.wikipedia.org/wiki/Portable_Document_Format
.. _Postscript: https://en.wikipedia.org/wiki/PostScript
.. _`Scalable Vector Graphics`: https://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _pgf: https://ctan.org/pkg/pgf
.. _Cairo: https://www.cairographics.org
.. _PyGObject: https://wiki.gnome.org/action/show/Projects/PyGObject
.. _pycairo: https://www.cairographics.org/pycairo/
.. _cairocffi: https://pythonhosted.org/cairocffi/
.. _wxPython: https://www.wxpython.org/
.. _TkInter: https://docs.python.org/3/library/tk.html
.. _PyQt: https://riverbankcomputing.com/software/pyqt/intro
.. _`Qt for Python`: https://doc.qt.io/qtforpython/
.. _Qt: https://qt.io/
.. _GTK: https://www.gtk.org/
.. _Tk: https://www.tcl.tk/
.. _wxWidgets: https://www.wxwidgets.org/
.. _ipympl: https://www.matplotlib.org/ipympl

ipympl
^^^^^^

The Jupyter widget ecosystem is moving too fast to support directly in
Matplotlib.  To install ipympl:

.. code-block:: bash

   pip install ipympl

or

.. code-block:: bash

   conda install ipympl -c conda-forge

See `installing ipympl <https://matplotlib.org/ipympl/installing.html>`__ for more details.

.. _QT_API-usage:

How do I select the Qt implementation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The QtAgg and QtCairo backends support both Qt 5 and 6, as well as both Python
bindings (`PyQt`_ or `Qt for Python`_, a.k.a. PySide). If any binding has
already been loaded, then it will be used for the Qt backend. Otherwise, the
first available binding is used, in the order: PyQt6, PySide6, PyQt5, PySide2.

The :envvar:`QT_API` environment variable can be set to override the search
when nothing has already been loaded. It may be set to (case-insensitively)
PyQt6, PySide6, PyQt5, or PySide2 to pick the version and binding to use. If
the chosen implementation is unavailable, the Qt backend will fail to load
without attempting any other Qt implementations.  See :ref:`QT_bindings` for
more details.

Using non-builtin backends
--------------------------
More generally, any importable backend can be selected by using any of the
methods above. If ``name.of.the.backend`` is the module containing the
backend, use ``module://name.of.the.backend`` as the backend name, e.g.
``matplotlib.use('module://name.of.the.backend')``.
