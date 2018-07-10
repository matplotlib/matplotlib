.. _interactive_figures_and_eventloops:

.. currentmodule:: matplotlib


==================================================
 Interactive Figures and Asynchronous Programming
==================================================

Matplotlib supports rich interactive figures by embedding figures into
a GUI window.  The basic interactions of panning and zooming in as
Axis to inspect your data is 'baked in' to Matplotlib.  This is
supported by a full mouse and keyboard event handling system to enable
you to build sophisticated interactive graphs.

This is meant to be an introduction to the low-level details of how
integrating the Matplotlib with a GUI event loop works.  For a more
practical introduction the Matplotlib event API see `Interactive
Tutorial <https://github.com/matplotlib/interactive_tutorial>`__ and
`Interactive Applications using Matplotlib
<http://www.amazon.com/Interactive-Applications-using-Matplotlib-Benjamin/dp/1783988843>`__.

Event Loops
===========

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
all terminals, GUIs, and servers [#f1]_.  In general the *Read* step
is waiting on some sort of I/O -- be it user input or the network --
while the *Evaluate* and *Print* are responsible for interpreting the
input and then **doing** something about it.

In practice users do not work directly with these loops and instead
use a framework that provides a mechanism to register callbacks to be
called in response to specific event.  For example "when the user
clicks on this button, please run this function" or "when the user
hits the 'z' key, please run this other function".  This allows users
to write reactive, event-driven, programs without having to delve into
the nity-gritty [#f2]_ details of I/O.  Examples of this pattern the
``Signal`` / ``Slot`` framework in Qt (and the analogs in other GUI
frameworks), the 'request' functions in flask, and Matplotlib's
:ref:`event handling system <event-handling-tutorial>`.

All GUI frameworks (Qt, Wx, Gtk, tk, OSX, or web) have some method of
capturing user interactions and passing them back to the application,
although the exact details vary.  Matplotlib has a :ref:`backnd
<what-is-a-backend>` for each GUI framework which use these mechanisms
to convert the native UI events into Matplotlib events so we can
develop framework-independent interactive figures.


references to trackdown:
 - link to cpython REPL loop (pythonrun.c::PyRunInteractiveLoopFlags)
 - link to IPython repl loop (Ipython.terminal.interactiveshell.py :: TerminalInteractiveShell.mainloop
   and Ipython.terminal.interactiveshell.py :: TerminalInteractiveShell.interact)
 - curio / trio / asycio / twisted / tornado event loops
 - Beazly talk or two on asyncio


.. _cp_integration:

Command Prompt Integration
==========================

Integrating a GUI window into a CLI introduces a conflict: there are
two infinite loops that want to be waiting for user input on the main
thread in the same process.  Python wants to be blocking the main
thread it it's REPL loop and the GUI framework wants to be running
it's event loop.  In order for both the prompt and the GUI widows to
be responsive we need a method to allow the loops to 'timeshare' :

1. let the GUI main loop block the python process when you want
   interactive windows
2. let the CLI main loop block the python process and intermittently
   run the GUI loop

The inverse problem, embedding an interactive prompt into a GUI, is
also a way out of this problem by letting the GUI event loop always
run the show, but that is essentially writing a full application.


Blocking the Prompt
-------------------

The simplest "integration" is to start the GUI event loop in
'blocking' mode and take over the CLI.  While the GUI event loop is
running you can not enter new commands into the prompt (your terminal
may show the charters entered into stdin, but they will not be
processed by python), but the figure windows will be responsive.  Once
the event loop is stopped (leaving any still open figure windows
non-responsive) you will be able to use the prompt again.  Re-starting
the event loop will make any open figure responsive again.


To start the event loop until all open figures are closed use
`pyplot.show(block=True)`.  To start the event loop for a fixed amount
of time use `pyplot.pause`.

Without using ``pyplot`` you can start and stop the event loops via
``fig.canvas.start_event_loop`` and ``fig.canvas.stop_event_loop``.


.. warning

   By using `Figure.show` it is possible to display a figure on the
   screen without explicitly starting the event loop and not being in
   interactive mode.  This may work but will likely result in a
   non-responsive figure and may not even display the rendered plot.


This technique can be very useful if you want to write a script that
pauses for user interaction, see :ref:`interactive_script`.

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.show
   pyplot.pause

   backend_bases.FigureCanvasBase.start_event_loop
   backend_bases.FigureCanvasBase.stop_event_loop

   figure.Figure.show


Explicitly running the Event Loop
---------------------------------

If you have open windows (either due to a `plt.pause` timing out or
from calling `figure.Figure.show`) that have pending UI events (mouse
clicks, button presses, or draws) you can explicitly process them by
calling ``fig.canvas.flush_events()``.  This will run the GUI event
loop, until all UI events currently waiting have been processed.  The
exact behavior is backend-dependent but typically events on all figure
are processed and only events waiting to be processed (not those added
during processing) will be handled.

For example ::

   import time
   import matplotlib.pyplot as plt
   import numpy as np
   plt.ion()

   fig, ax = plt.subplots()
   fig.canvas.show()
   th = np.linspace(0, 2*np.pi, 512)
   ax.set_ylim(-1.5, 1.5)

   ln, = ax.plot(th, np.sin(th))

   def slow_loop(N, ln):
       for j in range(N):
           time.sleep(.1)  # to simulate some work
           ln.figure.canvas.flush_events()

   slow_loop(100, ln)

Will be a bit laggy, as we are only processing user input every 100ms
(where as 20-30ms is what feels "responsive"), but it will respond.


If you make changes to the plot and want it re-rendered you will need
to call `~.FigureCanvasBase.draw_idle()` to request that the canvas be
re-drawn.  This method can be thought of *draw_soon* in analogy to
`asyncio.BaseEventLoop.call_soon`.

We can add this our example above as ::

   def slow_loop(N, ln):
       for j in range(N):
           time.sleep(.1)  # to simulate some work
           if j % 10:
               ln.set_ydata(np.sin(((j // 10) % 5 * th)))
               ln.figure.canvas.draw_idle()

           ln.figure.canvas.flush_events()

   slow_loop(100, ln)




.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.flush_events
   backend_bases.FigureCanvasBase.draw_idle


Interactive
-----------

While running the GUI event loop in a blocking mode or explicitly
handling UI events is useful, we can do better!  We really want to be
able to have a usable prompt **and** interactive figure windows.

We can do this using the 'input hook' feature of the interactive
prompt.  This hook is called by the prompt as it waits for the user
type (even for a fast typist the prompt is mostly waiting for the
human to think and move their fingers).  Although the details vary
between prompts the logic is roughly

1. start to wait for keyboard input
2. start the GUI event loop
3. as soon as the user hits a key, exit the GUI event loop and handle the key
4. repeat

This gives us the illusion of simultaneously having an interactive GUI
windows and an interactive prompt.  Most of the time the GUI event
loop is running, but as soon as the user starts typing the prompt
takes over again.

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


.. _interactive_scripts :

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
instances, calling ``draw_idle``, and flushing the GUI event loop with your
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
           while True:                    # Loop
               inp = select(fds).read()   # Read
               eval(inp)                  # Evaluate / Print


.. [#f2] These examples are agressively dropping many of the
	 complexities that must be dealt with in the real world such as
	 keyboard interupts [link], timeouts, bad input, resource
	 allocation and cleanup, etc.
