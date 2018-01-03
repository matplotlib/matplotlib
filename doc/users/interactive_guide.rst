.. _plotting-guide-interactive:

.. currentmodule:: matplotlib


==================================================
 Interactive Figures and Asynchronous Programming
==================================================

Matplotlib supports rich interactive figures.  At the most basic
matplotlib has the ability to zoom and pan a figure to inspect your
data 'baked in', but this is backed by a full mouse and keyboard event
handling system to enable users to build sophisticated interactive
graphs.

This is meant to be a rapid introduction to the relevant details of
integrating the matplotlib with a GUI event loop.  For further details
see `Interactive Applications using Matplotlib
<http://www.amazon.com/Interactive-Applications-using-Matplotlib-Benjamin/dp/1783988843>`__.

Fundamentally, all user interaction (and networking) is implemented as
an infinite loop waiting for events from the user (via the OS) and
then doing something about it.  For example, a minimal Read Evaluate
Print Loop (REPL) is ::

  exec_count = 0
  while True:
      inp = input(f"[{exec_count}] > ")        # Read
      ret = eval(inp)                          # Evaluate
      print(ret)                               # Print
      exec_count += 1                          # Loop


This is missing many niceties (for example, it exits on the first
exception!), but is representative of the event loops that underlie
all terminals, GUIs, and servers [#f1]_.  In general the *Read* step is
waiting on some sort of I/O, be it user input from a keyboard or mouse
or the network while the *Evaluate* and *Print* are responsible for
interpreting the input and then **doing** something about it.

In practice users do not work directly with these loops and instead
use a framework that provides a mechanism to register callbacks [#2]_.
This allows users to write reactive, event-driven, programs without
having to delve into the nity-grity [#f3]_ details of I/O.  Examples
include ``observe`` and friends in `traitlets`, the ``Signal`` /
``Slot`` framework in Qt (and the analogs in other GUI frameworks),
the 'request' functions in web frameworks, and Matplotlib's native
:ref:`event handling system <event-handling-tutorial>`.

All GUI frameworks (Qt, Wx, Gtk, tk, QSX, or web) have some method of
capturing user interactions and passing them back to the application,
although the exact details vary between frameworks.  Matplotlib has a
'backend' (see :ref:`what-is-a-backend`) for each GUI framework which
converts the native UI events into Matplotlib events.  This allows
Matplotlib user to write GUI-independent interactive figures.

references to trackdown:
 - link to cpython REPL loop (pythonrun.c::PyRunInteractiveLoopFlags)
 - link to IPython repl loop (Ipython.terminal.interactiveshell.py :: TerminalInteractiveShell.mainloop
   and Ipython.terminal.interactiveshell.py :: TerminalInteractiveShell.interact)
 - curio / trio / asycio / twisted / tornado event loops
 - Beazly talk or two on asyncio



Interactive Mode
================

.. _cp_integration:

Command Prompt Integration
==========================

Integrating a GUI window and a CLI introduces a conflict: there are
two infinite loops that want to be waiting for user input in the same
process.  In order for both the prompt and the GUI widows to be responsive
we need a method to allow the loops to 'timeshare'.

1. let the GUI main loop block the python process
2. intermittently run the GUI loop


Blocking
--------

The simplest "integration" is to start the GUI event loop in
'blocking' mode and take over the CLI.  While the GUI event loop is
running you can not enter new commands into the prompt (your terminal
may show the charters entered into stdin, but they will not be
processed by python), but the figure windows will be responsive.  Once
the event loop is stopped (leaving any still open figure windows
non-responsive) you will be able to use the prompt again.  Re-starting
the event will make any open figure responsive again.


To start the event loop until all open figures are closed us
`pyplot.show(block=True)`.  To start the event loop for a fixed amount
of time use `pyplot.pause`.

Without using ``pyplot`` you can start and stop the event loops via
``fig.canvas.start_event_loop`` and ``fig.canvas.stop_event_loop``.


.. warning

   By using `Figure.show` it is possible to display a figure on the
   screen with out starting the event loop and not being in
   interactive mode.  This will likely result in a non-responsive
   figure and may not even display the rendered plot.



.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.show
   pyplot.pause

   backend_bases.FigureCanvasBase.start_event_loop
   backend_bases.FigureCanvasBase.stop_event_loop

   figure.Figure.show


Explicitly
----------

If you have open windows (either due to a `plt.pause` timing out or
from calling `figure.Figure.show`) that have pending UI events (mouse
clicks, button presses, or draws) you can explicitly process them by
calling ``fig.canvas.flush_events()``.  This will not run the GUI
event loop, but instead synchronously processes all UI events current
waiting to be processed.  The exact behavior is backend-dependent but
typically events on all figure are processed and only events waiting
to be processed (not those added during processing) will be handled.

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.flush_events

Interactive
-----------

While running the GUI event loop in a blocking mode or explicitly
handling UI events is useful, we can do better!  We really want to be
able to have a usable prompt **and** interactive figure windows.  We
can do this using the concept of 'input hook' (see :ref:`below
<Eventloop integration mechanism>` for implementation details) that
allows the GUI event loop to run and process events while the prompt
is waiting for the user to type (even for an extremely fast typist, a
vast majority of the time the prompt is simple idle waiting for human
finders to move).  This effectively gives us a simultaneously GUI
windows and prompt.

This time-share technique only allows the event loop to run while
python is otherwise idle and waiting for user input.  If you want the
GUI to be responsive during long running code it is necessary to
periodically flush the GUI event queue as described :ref:`above
<Explicitly>`.  In this case it is your code, not the REPL, which is
blocking process so you need to handle the time-share manually.
Conversely, a very slow figure draw will block the prompt until it
finishes.

Full embedding
==============

It is also possible to go the other direction and fully embed figures
it a rich native application.  Matplotlib provides classes which can
be directly embedded in GUI applications (this is how the built-in
windows are implemented!).  See :ref:`user_interfaces` for more details
on how to do this.


Scripts
=======

There are several use-cases for using interactive figures in scripts:

- progress updates as a long running script progresses
- capture user input to steer the script
- streaming updates from a data source

In the first case, it is the same as :ref:`above
<Explicitly>` where you explicitly call ::

  fig.canvas.flush_events()

periodically to allow the event loop to process UI and draw events and
::

   fig.canvas.draw_idle()

when you have updated the contents of the figure.

The more frequently you call ``flush_events`` the more responsive your
figure will feel but at the cost of spending more resource on the
visualization and less on your computation.

The second case is very much like :ref:`Blocking` above.  By using ``plt.show(block=True)`` or

The third case you will have to integrate updating the ``Aritist``
instances, calling ``draw_idle``, and the GUI event loop with your
data I/O.

.. _stale_artists:

Stale Artists
=============

Artists (as of 1.5) have a ``stale`` attribute which is `True` if the
internal state of the artist has changed since the last time it was
drawn to the screen.  The stale state is propagated up to the Artists
parents in the draw tree.  Thus, ``fig.stale`` will report of any
artist in the figure has been modified and out of sync with what is
displayed on the screen.  This is intended to be used to determine if
``draw_idle`` should be called to schedule a re-rendering of the
figure.

TODO:

- notes about callbacks
-


Draw Idle
=========

In almost all cases, we recommend using
`backend_bases.FigureCanvasBae.draw_idle` over
`backend_bases.FigureCanvasBae.draw`.  ``draw`` forces a rendering of
the figure where as ``draw_idle`` schedules a rendering the next time
the GUI window is going to re-paint the screen.  This improves
performance by only rendering pixels that will be shown on the screen.  If
you want to be sure that the screen is updated as soon as possible do ::

  fig.canvas.draw_idle()
  fig.canvas.flush_events()



.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.draw
   backend_bases.FigureCanvasBase.draw_idle
   backend_bases.FigureCanvasBase.flush_events

Threading
=========

Unfortunately, most GUI frameworks require that all updates to the
screen happen on the main thread which makes pushing periodic updates
to the window to a background thread problematic.  Although it seems
backwards, it is easier to push your computations to a background
thread and periodically update the figure from the main thread.

In general Matplotlib is not thread safe, but the consequences of
drawing while updating artists from another thread should not be worse
than a failed draw.  This should not be fatal and so long as the
Artists end up consistent the figure can eventually be drawn cleanly.

Web
===



The Weeds
=========


Eventloop integration mechanism
-------------------------------

CPython / readline
~~~~~~~~~~~~~~~~~~

The python capi provides a hook, `PyOS_InputHook`, to register a
function to be run "The function will be called when Python's
interpreter prompt is about to become idle and wait for user input
from the terminal.".  This hook can be used to integrate a second
event loop (the GUI event loop) with the python input prompt loop.
The hook functions typically exhaust all pending events on the GUI
event queue, run the main loop for a short fixed amount of time, or
run the event loop until a key is pressed on stdin.


Matplotlib does not currently do any management of `PyOS_InputHook`
due to the wide range of ways that matplotlib is used.  This
management is left to the code using Matplotlib.  Interactive figures,
even with matplotlib in 'interactive mode', may not work in the
vanilla python repl if an appropriate `PyOS_InputHook` is not
registered.

Input hooks, and helpers to install them, are usually included with
the python bindings for GUI toolkits and may be registered on import.
IPython also ships input hook functions for all of the GUI frameworks
Matplotlib supports which can be installed via ``%matplotlib``.  This
is the recommended method of integrating Matplotlib and a prompt.



IPython / prompt toolkit
~~~~~~~~~~~~~~~~~~~~~~~~

With IPython >= 5.0 IPython has changed from using cpython's readline
based prompt to a ``prompt_toolkit`` based prompt.  ``prompt_toolkit``
has the same conceptual input hook, which is feed into pt via the
:meth:`IPython.terminal.interactiveshell.TerminalInteractiveShell.inputhook`
method.  The source for the prompt_toolkit input hooks lives at
:mod:`IPython.terminal.pt_inputhooks`



.. rubric:: Fotenotes

.. [#f1] A limitation of this design is that you can only wait for one
	 input, if there is a need to multiplex between multiple sources
	 then the loop would look something like ::

	   fds = [...]
           while True:             # Loop
               inp = select(fds)   # Read
               eval(inp)           # Evaluate / Print


.. [#f2] asyncio has a fundamentally different paradigm that uses
         coroutines instead of callbacks as the user-facing interface,
         however at the core there is a select loop like the above
	 footnote the multiplexes between the running tasks.

.. [#f3] These examples are agressively dropping many of the
	 complexities that must be dealt with in the real world such as
	 keyboard interupts [link], timeouts, bad input, resource
	 allocation and cleanup, etc.
