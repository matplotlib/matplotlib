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
can be an expensive operation, and you do not want to re-render the
figure every time a single property is changed, only once after all
the properties have changed and the figure is displayed on the screen
or saved to disk.  When working in a shell, one typically wants the
figure to re-draw after every user command (ex when the prompt comes
back).  *interactive* mode of :mod:`~.pyplot` takes care of arranging
such that if any open figures are :ref:`stale <stale_artists>`, they
will be re-drawn just before the prompt is returned to the user.

.. _ipython-pylab:

IPython to the rescue
=====================

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

Other python interpreters
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
