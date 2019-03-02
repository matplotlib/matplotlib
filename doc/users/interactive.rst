.. currentmodule:: matplotlib

.. _mpl-shell:

===================
 Interactive plots
===================

.. toctree::
    :maxdepth: 2

    interactive_guide
    event_handling.rst


By default, matplotlib defers drawing until explicitly asked.  Drawing
may be an expensive operation and some changes to the figure may leave
it in an inconsistent state, hence we only want to render the figure
just before the figure is displayed on the screen or saved to disk.
In the case of scripts this is easy to arrange, we simple do nothing
until ``savefig`` is called.  However, when working interactively,
either at a terminal or in a notebook, we would like the figure to
automatically re-render when we change.  Further, we would like the
figures respond to user input (mouse and keyboard) to interact with
and explore the data.

To get this functionality we need a combination of an interactive
backend (to handle the user interaction with the figure) and an
interactive interpreter (to handle the input of user code).  Together
these tools need to:

- Take text from the user and execute it in the Python interpreter
- Ensure that any created figures are put on the screen
- Trigger a re-render of the figure when the user has mutated it
- When not actively executing user input, run the event loop so we can
  get user input on the figure (see
  :ref:`interactive_figures_and_eventloops` for details)



.. _ipython-pylab:

IPython to the rescue!
======================

To support interactive

We recommend using IPython for an interactive shell.  In addition to
all of it's features (improved tab-completion, magics,
multiline-editing, etc), it also ensures that the GUI toolkit is
properly integrated with the command line (see :ref:`cp_integration`).
To configure the integration and enable interactive mode do::

.. sourcecode:: ipython

   user@machine:~ $ ipython
   Python 3.6.4 (default, Dec 23 2017, 19:07:07)
   Type 'copyright', 'credits' or 'license' for more information
   IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

   In [1]: %matplotlib
   iUsing matplotlib backend: Qt5Agg

   In [2]: import matplotlib.pyplot as plt

   In [3]:

Calling

.. sourcecode:: ipython

   In [3]: fig, ax = plt.subplots()

will pop open a window for you and

.. sourcecode:: ipython

   In [4]: ln, = ax.plot(range(5))

will show your data in the window.  If you
want to change anything about the line, ex the color

.. sourcecode:: ipython

   In [5]: ln.set_color('orange')

will be reflected immediately.



.. _other-shells:

Other Python interpreters
=========================

If you can not or do not want to use IPython, interactive mode
should work in the vanilla python prompt


.. sourcecode:: python

   user@machine:~ $ python
   Python 3.6.4 (default, Dec 23 2017, 19:07:07)
   [GCC 7.2.1 20171128] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import matplotlib.pyplot as plt
   >>> plt.ion()
   >>>

however this does not ensure that the event hook is properly installed
and your figures may not be responsive (see :ref:`cp_integration`).

GUI python prompts may or may not work with all backends.  For
example, if you use the IDLE IDE you must use the 'tkagg' backend (as
IDLE is a Tk application..  However, if you use spyder, the
interactive shell is run in a sub process of the GUI and you can use
any backend.


PyCharm
-------

TODO: links to pycharm docs on configuring Matplotlib backends

Spyder
------

TODO: links to spyder docs on configuring Matplotlib backends

VSCode
------

TODO: links to vscode docs on configuring Matplotlib backends (?)


Jupyter Notebooks / Jupyter Lab
-------------------------------

.. warning::

   To get the interactive functionality described here, you must be
   using an interactive backend, however he 'inline' backend is not.
   It renders the figure once and inserts a static image into the
   notebook everytime the cell in run.  Being static these plots can
   not be panned / zoomed or take user input.

Jupyter uses a different architecture than a traditional interactive
terminal.  Rather than the user interface running in the same process
as your interpreter, the user interacts with a javascript front end
running in a browser which communicates with a server which in turn
communicates with a kernel that actually runs the python.  This means
for interactive figures our UI needs to also be written in javascript
and run inside of the jupyter front end.

To get interactive figures in the 'classic' notebook or jupyter lab use
the `ipympl` backend (must be installed separately) which uses the `ipywidget`
framework and which enabled by using ::

  %matplotlib widget

If you only need to use the classic notebook you can also use ::

  %matplotlib notebook

which uses the `nbagg` backend which ships with Matplotlib (but does
not work with JLab because, for security reasons, you can no longer
inject arbitrary javascript into the front end).

GUIs + jupyter
~~~~~~~~~~~~~~

If you are running your jupyter server locally you can use one of the
GUI backends, however if you ever move that notebook to a remote
server it will cease to work correctly.  What is happening is that
when you create a figure the process running your kernel creates and
shows a GUI window.  If that process is on the same computer as you,
then you will see it, however if it is running on a remote computer it
will try to open the GUI window on _that_ computer.  This will either
fail by raising an exception (as many linux servers do not have an
XServer running) or run cleanly, but leave you with no way to access
your figure.




.. _controlling-interactive:

Controlling interactive updating
================================


To control and query the current state of *interactive* mode

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.ion
   pyplot.ioff
   pyplot.isinteractive

When working with a big figure in which drawing is expensive, you may
want to turn matplotlib's interactive setting off temporarily to avoid
the performance hit::


    >>> #create big-expensive-figure
    >>> plt.ioff()      # turn updates off
    >>> ax.set_title('now how much would you pay?')
    >>> fig.canvas.draw_idle()      # force a draw
    >>> fig.savefig('alldone', dpi=300)
    >>> plt.close('all')
    >>> plt.ion()      # turn updating back on
    >>> fig, ax = plt.subplots()
    >>> ax.plot(rand(20), mfc='g', mec='r', ms=40, mew=4, ls='--', lw=3)


Default UI
==========


.. toctree::
    :maxdepth: 1

    navigation_toolbar.rst


The windows created by :mod:`~.pyplot` have an interactive toolbar and
has a number of helpful keybindings by default.

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
