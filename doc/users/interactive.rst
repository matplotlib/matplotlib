.. currentmodule:: matplotlib

.. _mpl-shell:

=====================
 Interactive Figures
=====================

.. toctree::


When working with data it is often invaluable to be able to interact
with your plots In many cases the built in pan/zoom and mouse-location
tools are sufficient, but you can also use the Matplotlib event system
to build a customized data exploration tools.

Matplotlib ships with :ref:`backends <what-is-a-backend>` binding to
several GUI toolkits (Qt, Tk, Wx, Gtk, OSX, js) and third party
packages provide bindings to `kivy
<https://github.com/kivy-garden/garden.matplotlib>`__ and `Jupyter Lab
<https://github.com/matplotlib/ipympl>`__.  For the figures to be
"live" the GUI event loop will need to be integrated with your prompt, the
simplest way is to use IPython (see :ref:`below <ipython-pylab>`).

The `.pyplot` module provides two functions for creating Figures that,
when an interactive backend is used, that are managed by Matplotlib,
embedded in GUI windows, and ready for interactive use out of the box

`.pyplot.figure`
    Creates a new empty `.figure.Figure` or selects an existing figure

`.pyplot.subplots`
    Creates a new `.figure.Figure` and fills it with a grid of `.axes.Axes`

Matplotlib keeps a reference to all of the open figures created this
way so they will not be garbage collected.  You can close all off your
open figures via ``plt.close('all')``.

For discussion of how this works under the hood see:

.. toctree::
    :maxdepth: 1

    interactive_guide.rst
    event_handling.rst


.. _ipython-pylab:

IPython integration
===================

We recommend using IPython for an interactive shell.  In addition to
all of its features (improved tab-completion, magics,
multiline-editing, etc), it also ensures that the GUI toolkit event
loop is properly integrated with the command line (see
:ref:`cp_integration`).  To configure the integration and enable
:ref:`interactive mode <controlling-interactive>` use the
``%matplotlib`` magic

.. highlight:: ipython

::

   user@machine:~ $ ipython
   Python 3.8.2 (default, Apr  8 2020, 14:31:25)
   Type 'copyright', 'credits' or 'license' for more information
   IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.

   In [1]: %matplotlib
   Using matplotlib backend: Qt5Agg

   In [2]: import matplotlib.pyplot as plt

   In [3]:

Calling

::

   In [3]: fig, ax = plt.subplots()

will pop open a window for you and

::

   In [4]: ln, = ax.plot(range(5))

will show your data in the window.  If you change something about the
line, for example the color

::

   In [5]: ln.set_color('orange')

it will be reflected immediately. If you wish to disable this behavior
use

::

   In [6]: plt.ioff()

and

::

   In [7]: plt.ion()

re-enable it.

With recent versions of ``Matplotlib`` and ``IPython`` it is
sufficient to import `matplotlib.pyplot` and call `.pyplot.ion`, but
using the magic is guaranteed to work regardless of versions.


.. highlight:: python

.. _controlling-interactive:

Interactive mode
================


.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.ion
   pyplot.ioff
   pyplot.isinteractive


.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.show
   pyplot.pause


Interactive mode controls:

- if created figures are automatically shown
- if changes to artists automatically trigger re-drawing existing figures
- if `.pyplot.show` blocks or not


if in interactive mode then:

- newly created figures will be shown immediately
- figures will automatically redraw on change
- pyplot.show will not block by default

If not in interactive mode then:

- newly created figures and changes to figures will
  not be reflected until explicitly asked to be
- pyplot.show will block by default


If you are in non-interactive mode (or created figures while in
non-interactive mode) you may need to explicitly call `.pyplot.show`
to bring the windows onto your screen.  If you only want to run the
GUI event loop for a fixed amount of time you can use `.pyplot.pause`.

Being in interactive mode is orthogonal to the GUI event loop being
integrated with your command prompt.  If you have the GUI event loop
integrated with your prompt, then shown figures will be "live" while
the prompt is waiting for input, if it is not integrated than your
figures will only be "live" when the GUI event loop is running (via
`.pyplot.show`, `.pyplot.pause`, or explicitly starting the GUI main
loop).


.. warning

   Using `.figure.Figure.show` it is possible to display a figure on
   the screen without starting the event loop and without being in
   interactive mode.  This may work (depending on the GUI toolkit) but
   will likely result in a non-responsive figure.

.. _navigation-toolbar:

Default UI
==========


The windows created by :mod:`~.pyplot` have an interactive toolbar with navigation
buttons and a readout of where the cursor is in dataspace.  A number of
helpful keybindings are registered by default.


.. _key-event-handling:

Navigation Keyboard Shortcuts
-----------------------------

The following table holds all the default keys, which can be
overwritten by use of your matplotlibrc (#keymap.\*).

================================== =================================================
Command                            Keyboard Shortcut(s)
================================== =================================================
Home/Reset                         **h** or **r** or **home**
Back                               **c** or **left arrow** or **backspace**
Forward                            **v** or **right arrow**
Pan/Zoom                           **p**
Zoom-to-rect                       **o**
Save                               **ctrl** + **s**
Toggle fullscreen                  **f** or **ctrl** + **f**
Close plot                         **ctrl** + **w**
Close all plots                    **shift** + **w**
Constrain pan/zoom to x axis       hold **x** when panning/zooming with mouse
Constrain pan/zoom to y axis       hold **y** when panning/zooming with mouse
Preserve aspect ratio              hold **CONTROL** when panning/zooming with mouse
Toggle major grids                 **g** when mouse is over an axes
Toggle minor grids                 **G** when mouse is over an axes
Toggle x axis scale (log/linear)   **L** or **k**  when mouse is over an axes
Toggle y axis scale (log/linear)   **l** when mouse is over an axes
================================== =================================================


.. _other-shells:

Other Python prompts
====================

If you can not or do not want to use IPython, interactive mode works
in the vanilla python prompt


.. sourcecode:: pycon

   >>> import matplotlib.pyplot as plt
   >>> plt.ion()
   >>>

however this does not ensure that the event hook is properly installed
and your figures may not be responsive.  Please consult the
documentation of your GUI toolkit for details.



Jupyter Notebooks / Lab
-----------------------

.. note::

   To get the interactive functionality described here, you must be
   using an interactive backend.  The default backend in notebooks,
   the inline backend, is not.  `~ipykernel.pylab.backend_inline`
   renders the figure once and inserts a static image into the
   notebook when the cell is executed.  The images are static and can
   not be panned / zoomed, take user input, or be updated from other
   cells.

To get interactive figures in the 'classic' notebook or jupyter lab
use the `ipympl <https://github.com/matplotlib/ipympl>`__ backend
(must be installed separately) which uses the **ipywidget** framework.
If ``ipympl`` is installed use the magic:

.. sourcecode:: ipython

  %matplotlib widget

to select and enable it.

If you only need to use the classic notebook you can use

.. sourcecode:: ipython

  %matplotlib notebook

which uses the `.backend_nbagg` backend which ships with Matplotlib.
However nbagg does not work in Jupyter Lab due to changes in the front
end.

GUIs + jupyter
~~~~~~~~~~~~~~

If you are running your jupyter server locally you can use one of the
GUI backends.  However if you ever move that notebook to a remote
server it will cease to work correctly because the GUI windows will be
created on the server, not your machine. When you create a figure the
process running your kernel creates and shows a GUI window.  If that
process is on the same computer as your client, then you will be able
to see and interact with the window.  However if it is running on a
remote computer it will try to open the GUI window on _that_ computer.
This will either fail by raising an exception (as many servers
do not have an XServer running) or run, but leave you with no
way to access your figure.



PyCharm, Spydter, and VSCode
----------------------------

Many IDEs have built-in integration with Matplotlib, please consult its
documentation for configuration details.
