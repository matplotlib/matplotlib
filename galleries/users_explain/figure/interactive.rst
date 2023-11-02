.. redirect-from:: /users/interactive
.. redirect-from:: /users/explain/interactive

.. currentmodule:: matplotlib

.. _mpl-shell:
.. _interactive_figures:

===================
Interactive figures
===================

When working with data, interactivity can be invaluable. The pan/zoom and
mouse-location tools built into the Matplotlib GUI windows are often sufficient, but
you can also use the event system to build customized data exploration tools.

.. seealso::
    :ref:`figure-intro`.


Matplotlib ships with :ref:`backends <what-is-a-backend>` binding to
several GUI toolkits (Qt, Tk, Wx, GTK, macOS, JavaScript) and third party
packages provide bindings to `kivy
<https://github.com/kivy-garden/garden.matplotlib>`__ and `Jupyter Lab
<https://matplotlib.org/ipympl>`__.  For the figures to be responsive to
mouse, keyboard, and paint events, the GUI event loop needs to be integrated
with an interactive prompt. We recommend using IPython (see :ref:`below <ipython-pylab>`).

The `.pyplot` module provides functions for explicitly creating figures
that include interactive tools, a toolbar, a tool-tip, and
:ref:`key bindings <key-event-handling>`:

`.pyplot.figure`
    Creates a new empty `.Figure` or selects an existing figure

`.pyplot.subplots`
    Creates a new `.Figure` and fills it with a grid of `~.axes.Axes`

`.pyplot.gcf`
    Get the current `.Figure`. If there is current no figure on the pyplot figure
    stack, a new figure is created

`.pyplot.gca`
    Get the current `~.axes.Axes`. If there is current no Axes on the Figure,
    a new one is created

Almost all of the functions in `.pyplot` pass through the current `.Figure` / `~.axes.Axes`
(or create one) as appropriate.

Matplotlib keeps a reference to all of the open figures
created via `pyplot.figure` or `pyplot.subplots` so that the figures will not be garbage
collected. `.Figure`\s can be closed and deregistered from `.pyplot` individually via
`.pyplot.close`; all open `.Figure`\s can be closed via ``plt.close('all')``.


.. seealso::

   For more discussion of Matplotlib's event system and integrated event loops:
   - :ref:`interactive_figures_and_eventloops`
   - :ref:`event-handling`


.. _ipython-pylab:

IPython integration
===================

We recommend using IPython for an interactive shell.  In addition to
all of its features (improved tab-completion, magics, multiline editing, etc),
it also ensures that the GUI toolkit event loop is properly integrated
with the command line (see :ref:`cp_integration`).

In this example, we create and modify a figure via an IPython prompt.
The figure displays in a QtAgg GUI window. To configure the integration
and enable :ref:`interactive mode <controlling-interactive>` use the
``%matplotlib`` magic:

.. highlight:: ipython

::

   In [1]: %matplotlib
   Using matplotlib backend: QtAgg

   In [2]: import matplotlib.pyplot as plt

Create a new figure window:

::

   In [3]: fig, ax = plt.subplots()


Add a line plot of the data to the window:

::

   In [4]: ln, = ax.plot(range(5))

Change the color of the line from blue to orange:

::

   In [5]: ln.set_color('orange')

If you wish to disable automatic redrawing of the plot:

::

   In [6]: plt.ioff()

If you wish to re-enable automatic redrawing of the plot:

::

   In [7]: plt.ion()


In recent versions of ``Matplotlib`` and ``IPython``, it is
sufficient to import `matplotlib.pyplot` and call `.pyplot.ion`.
Using the ``%`` magic is guaranteed to work in all versions of Matplotlib and IPython.


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

- whether created figures are automatically shown
- whether changes to artists automatically trigger re-drawing existing figures
- when `.pyplot.show()` returns if given no arguments: immediately, or after all of the figures have been closed

If in interactive mode:

- newly created figures will be displayed immediately
- figures will automatically redraw when elements are changed
- `pyplot.show()` displays the figures and immediately returns

If not in interactive mode:

- newly created figures and changes to figures are not displayed until

  * `.pyplot.show()` is called
  * `.pyplot.pause()` is called
  * `.FigureCanvasBase.flush_events()` is called

- `pyplot.show()` runs the GUI event loop and does not return until all the plot windows are closed

If you are in non-interactive mode (or created figures while in
non-interactive mode) you may need to explicitly call `.pyplot.show`
to display the windows on your screen.  If you only want to run the
GUI event loop for a fixed amount of time, you can use `.pyplot.pause`.
This will block the progress of your code as if you had called
`time.sleep`, ensure the current window is shown and re-drawn if needed,
and run the GUI event loop for the specified period of time.

The GUI event loop being integrated with your command prompt and
the figures being in interactive mode are independent of each other.
If you try to use `pyplot.ion` without arranging for the event-loop integration,
your figures will appear but will not be interactive while the prompt is waiting for input.
You will not be able to pan/zoom and the figure may not even render
(the window might appear black, transparent, or as a snapshot of the
desktop under it).  Conversely, if you configure the event loop
integration, displayed figures will be responsive while waiting for input
at the prompt, regardless of pyplot's "interactive mode".

No matter what combination of interactive mode setting and event loop integration,
figures will be responsive if you use ``pyplot.show(block=True)``, `.pyplot.pause`, or run
the GUI main loop in some other way.


.. warning::

   Using `.Figure.show`, it is possible to display a figure on
   the screen without starting the event loop and without being in
   interactive mode.  This may work (depending on the GUI toolkit) but
   will likely result in a non-responsive figure.


.. _default_ui:

Default UI
==========

The windows created by :mod:`~.pyplot` have an interactive toolbar with navigation
buttons and a readout of the data values the cursor is pointing at.

.. _navigation-toolbar:

Interactive navigation
======================

.. image:: ../../../_static/toolbar.png

All figure windows come with a navigation toolbar, which can be used
to navigate through the data set.

.. image:: ../../../../lib/matplotlib/mpl-data/images/home_large.png

.. image:: ../../../../lib/matplotlib/mpl-data/images/back_large.png

.. image:: ../../../../lib/matplotlib/mpl-data/images/forward_large.png

The ``Home``, ``Forward`` and ``Back`` buttons
    These are similar to a web browser's home, forward and back controls.
    ``Forward`` and ``Back`` are used to navigate back and forth between
    previously defined views.  They have no meaning unless you have already
    navigated somewhere else using the pan and zoom buttons.  This is analogous
    to trying to click ``Back`` on your web browser before visiting a
    new page or ``Forward`` before you have gone back to a page --
    nothing happens.  ``Home`` takes you to the
    first, default view of your data.

.. image:: ../../../../lib/matplotlib/mpl-data/images/move_large.png

The ``Pan/Zoom`` button
    This button has two modes: pan and zoom.  Click the ``Pan/Zoom`` button
    to activate panning and zooming, then put your mouse somewhere
    over an axes.  Press the left mouse button and hold it to pan the
    figure, dragging it to a new position.  When you release it, the
    data under the point where you pressed will be moved to the point
    where you released.  If you press 'x' or 'y' while panning the
    motion will be constrained to the x or y axis, respectively.  Press
    the right mouse button to zoom, dragging it to a new position.
    The x axis will be zoomed in proportionately to the rightward
    movement and zoomed out proportionately to the leftward movement.
    The same is true for the y axis and up/down motions (up zooms in, down zooms out).
    The point under your mouse when you begin the zoom remains stationary, allowing you to
    zoom in or out around that point as much as you wish.  You can use the
    modifier keys 'x', 'y' or 'CONTROL' to constrain the zoom to the x
    axis, the y axis, or aspect ratio preserve, respectively.

    With polar plots, the pan and zoom functionality behaves
    differently.  The radius axis labels can be dragged using the left
    mouse button.  The radius scale can be zoomed in and out using the
    right mouse button.

.. image:: ../../../../lib/matplotlib/mpl-data/images/zoom_to_rect_large.png

The ``Zoom-to-Rectangle`` button
    Put your mouse somewhere over an axes and press a mouse button.  Define a rectangular region by
    dragging the mouse while holding the button to a new location.  When using
    the left mouse button, the axes view limits will be zoomed to the defined
    region.  When using the right mouse button, the axes view limits will be
    zoomed out, placing the original axes in the defined region.

.. image:: ../../../../lib/matplotlib/mpl-data/images/subplots_large.png

The ``Subplot-configuration`` button
    Use this button to configure the appearance of the subplot.
    You can stretch or compress the left, right, top, or bottom
    side of the subplot, or the space between the rows or
    space between the columns.

.. image:: ../../../../lib/matplotlib/mpl-data/images/filesave_large.png

The ``Save`` button
    Click this button to launch a file save dialog.  You can save
    files with the following extensions: ``png``, ``ps``, ``eps``,
    ``svg`` and ``pdf``.


.. _key-event-handling:

Navigation keyboard shortcuts
-----------------------------

A number of helpful keybindings are registered by default.  The following table
holds all the default keys, which can be overwritten by use of your
:ref:`matplotlibrc <customizing>`.

================================== ===============================
Command                            Default key binding and rcParam
================================== ===============================
Home/Reset                         :rc:`keymap.home`
Back                               :rc:`keymap.back`
Forward                            :rc:`keymap.forward`
Pan/Zoom                           :rc:`keymap.pan`
Zoom-to-rect                       :rc:`keymap.zoom`
Save                               :rc:`keymap.save`
Toggle fullscreen                  :rc:`keymap.fullscreen`
Toggle major grids                 :rc:`keymap.grid`
Toggle minor grids                 :rc:`keymap.grid_minor`
Toggle x axis scale (log/linear)   :rc:`keymap.xscale`
Toggle y axis scale (log/linear)   :rc:`keymap.yscale`
Close Figure                       :rc:`keymap.quit`
Constrain pan/zoom to x axis       hold **x** when panning/zooming with mouse
Constrain pan/zoom to y axis       hold **y** when panning/zooming with mouse
Preserve aspect ratio              hold **CONTROL** when panning/zooming with mouse
================================== ===============================


.. _other-shells:

Other Python prompts
====================

Interactive mode works in the default Python prompt:


.. sourcecode:: pycon

   >>> import matplotlib.pyplot as plt
   >>> plt.ion()
   >>>

However, this does not ensure that the event hook is properly installed
and your figures may not be responsive.  Please consult the
documentation of your GUI toolkit for details.


.. _jupyter_notebooks_jupyterlab:

Jupyter Notebooks / JupyterLab
------------------------------

To get interactive figures in the 'classic' notebook or Jupyter lab,
use the `ipympl <https://matplotlib.org/ipympl>`__ backend
(must be installed separately) which uses the **ipywidget** framework.
If ``ipympl`` is installed use the magic:

.. sourcecode:: ipython

  %matplotlib widget

to select and enable it.

If you only need to use the classic notebook (i.e. ``notebook<7``), you can use

.. sourcecode:: ipython

  %matplotlib notebook

which uses the `.backend_nbagg` backend provided by Matplotlib;
however, nbagg does not work in Jupyter Lab.

.. note::

   To get the interactive functionality described here, you must be
   using an interactive backend.  The default backend in notebooks,
   the inline backend, is not.  `~ipykernel.pylab.backend_inline`
   renders the figure once and inserts a static image into the
   notebook when the cell is executed.  Because the images are static, they
   cannot be panned / zoomed, take user input, or be updated from other
   cells.

GUIs + Jupyter
^^^^^^^^^^^^^^

You can also use one of the non-``ipympl`` GUI backends in a Jupyter Notebook.
If you are running your Jupyter kernel locally, the GUI window will spawn on
your desktop adjacent to your web browser. If you run your notebook on a remote server,
the kernel will try to open the GUI window on the remote computer. Unless you have
arranged to forward the xserver back to your desktop, you will not be able to
see or interact with the window. It may also raise an exception.



PyCharm, Spyder, and VSCode
---------------------------

Many IDEs have built-in integration with Matplotlib, please consult their
documentation for configuration details.
