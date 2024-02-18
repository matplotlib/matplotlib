.. _interactive_figures_and_eventloops:

.. redirect-from:: /users/interactive_guide

.. currentmodule:: matplotlib


================================================
Interactive figures and asynchronous programming
================================================

Matplotlib supports rich interactive figures by embedding figures into
a GUI window.  The basic interactions of panning and zooming in an
Axes to inspect your data is 'baked in' to Matplotlib.  This is
supported by a full mouse and keyboard event handling system that
you can use to build sophisticated interactive graphs.

This guide is meant to be an introduction to the low-level details of
how Matplotlib integration with a GUI event loop works.  For a more
practical introduction to the Matplotlib event API see :ref:`event
handling system <event-handling>`, `Interactive Tutorial
<https://github.com/matplotlib/interactive_tutorial>`__, and
`Interactive Applications using Matplotlib
<http://www.amazon.com/Interactive-Applications-using-Matplotlib-Benjamin/dp/1783988843>`__.

Event loops
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

In practice we interact with a framework that provides a mechanism to
register callbacks to be run in response to specific events rather
than directly implement the I/O loop [#f2]_.  For example "when the
user clicks on this button, please run this function" or "when the
user hits the 'z' key, please run this other function".  This allows
users to write reactive, event-driven, programs without having to
delve into the nitty-gritty [#f3]_ details of I/O.  The core event loop
is sometimes referred to as "the main loop" and is typically started,
depending on the library, by methods with names like ``_exec``,
``run``, or ``start``.


All GUI frameworks (Qt, Wx, Gtk, tk, macOS, or web) have some method of
capturing user interactions and passing them back to the application
(for example ``Signal`` / ``Slot`` framework in Qt) but the exact
details depend on the toolkit.  Matplotlib has a :ref:`backend
<what-is-a-backend>` for each GUI toolkit we support which uses the
toolkit API to bridge the toolkit UI events into Matplotlib's :ref:`event
handling system <event-handling>`.  You can then use
`.FigureCanvasBase.mpl_connect` to connect your function to
Matplotlib's event handling system.  This allows you to directly
interact with your data and write GUI toolkit agnostic user
interfaces.


.. _cp_integration:

Command prompt integration
==========================

So far, so good.  We have the REPL (like the IPython terminal) that
lets us interactively send code to the interpreter and get results
back.  We also have the GUI toolkit that runs an event loop waiting
for user input and lets us register functions to be run when that
happens.  However, if we want to do both we have a problem: the prompt
and the GUI event loop are both infinite loops that each think *they*
are in charge!  In order for both the prompt and the GUI windows to be
responsive we need a method to allow the loops to 'timeshare' :

1. let the GUI main loop block the python process when you want
   interactive windows
2. let the CLI main loop block the python process and intermittently
   run the GUI loop
3. fully embed python in the GUI (but this is basically writing a full
   application)

.. _cp_block_the_prompt:

Blocking the prompt
-------------------

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   pyplot.show
   pyplot.pause

   backend_bases.FigureCanvasBase.start_event_loop
   backend_bases.FigureCanvasBase.stop_event_loop


The simplest "integration" is to start the GUI event loop in
'blocking' mode and take over the CLI.  While the GUI event loop is
running you cannot enter new commands into the prompt (your terminal
may echo the characters typed into the terminal, but they will not be
sent to the Python interpreter because it is busy running the GUI
event loop), but the figure windows will be responsive.  Once the
event loop is stopped (leaving any still open figure windows
non-responsive) you will be able to use the prompt again.  Re-starting
the event loop will make any open figure responsive again (and will
process any queued up user interaction).

To start the event loop until all open figures are closed, use
`.pyplot.show` as ::

  pyplot.show(block=True)

To start the event loop for a fixed amount of time (in seconds) use
`.pyplot.pause`.

If you are not using `.pyplot` you can start and stop the event loops
via `.FigureCanvasBase.start_event_loop` and
`.FigureCanvasBase.stop_event_loop`. However, in most contexts where
you would not be using `.pyplot` you are embedding Matplotlib in a
large GUI application and the GUI event loop should already be running
for the application.

Away from the prompt, this technique can be very useful if you want to
write a script that pauses for user interaction, or displays a figure
between polling for additional data.  See :ref:`interactive_scripts`
for more details.


Input hook integration
----------------------

While running the GUI event loop in a blocking mode or explicitly
handling UI events is useful, we can do better!  We really want to be
able to have a usable prompt **and** interactive figure windows.

We can do this using the 'input hook' feature of the interactive
prompt.  This hook is called by the prompt as it waits for the user
to type (even for a fast typist the prompt is mostly waiting for the
human to think and move their fingers).  Although the details vary
between prompts the logic is roughly

1. start to wait for keyboard input
2. start the GUI event loop
3. as soon as the user hits a key, exit the GUI event loop and handle the key
4. repeat

This gives us the illusion of simultaneously having interactive GUI
windows and an interactive prompt.  Most of the time the GUI event
loop is running, but as soon as the user starts typing the prompt
takes over again.

This time-share technique only allows the event loop to run while
python is otherwise idle and waiting for user input.  If you want the
GUI to be responsive during long running code it is necessary to
periodically flush the GUI event queue as described in :ref:`spin_event_loop`.
In this case it is your code, not the REPL, which
is blocking the process so you need to handle the "time-share" manually.
Conversely, a very slow figure draw will block the prompt until it
finishes drawing.

Full embedding
==============

It is also possible to go the other direction and fully embed figures
(and a `Python interpreter
<https://docs.python.org/3/extending/embedding.html>`__) in a rich
native application.  Matplotlib provides classes for each toolkit
which can be directly embedded in GUI applications (this is how the
built-in windows are implemented!).  See :ref:`user_interfaces` for
more details.


.. _interactive_scripts :

Scripts and functions
=====================


.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.flush_events
   backend_bases.FigureCanvasBase.draw_idle

   figure.Figure.ginput
   pyplot.ginput

   pyplot.show
   pyplot.pause

There are several use-cases for using interactive figures in scripts:

- capture user input to steer the script
- progress updates as a long running script progresses
- streaming updates from a data source

Blocking functions
------------------

If you only need to collect points in an Axes you can use
`.Figure.ginput`.  However if you have written some custom event
handling or are using `.widgets` you will need to manually run the GUI
event loop using the methods described :ref:`above <cp_block_the_prompt>`.

You can also use the methods described in :ref:`cp_block_the_prompt`
to suspend run the GUI event loop.  Once the loop exits your code will
resume.  In general, any place you would use `time.sleep` you can use
`.pyplot.pause` instead with the added benefit of interactive figures.

For example, if you want to poll for data you could use something like ::

  fig, ax = plt.subplots()
  ln, = ax.plot([], [])

  while True:
      x, y = get_new_data()
      ln.set_data(x, y)
      plt.pause(1)

which would poll for new data and update the figure at 1Hz.

.. _spin_event_loop:

Explicitly spinning the event Loop
----------------------------------

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.flush_events
   backend_bases.FigureCanvasBase.draw_idle



If you have open windows that have pending UI
events (mouse clicks, button presses, or draws) you can explicitly
process those events by calling `.FigureCanvasBase.flush_events`.
This will run the GUI event loop until all UI events currently waiting
have been processed.  The exact behavior is backend-dependent but
typically events on all figure are processed and only events waiting
to be processed (not those added during processing) will be handled.

For example ::

   import time
   import matplotlib.pyplot as plt
   import numpy as np
   plt.ion()

   fig, ax = plt.subplots()
   th = np.linspace(0, 2*np.pi, 512)
   ax.set_ylim(-1.5, 1.5)

   ln, = ax.plot(th, np.sin(th))

   def slow_loop(N, ln):
       for j in range(N):
           time.sleep(.1)  # to simulate some work
           ln.figure.canvas.flush_events()

   slow_loop(100, ln)

While this will feel a bit laggy (as we are only processing user input
every 100ms whereas 20-30ms is what feels "responsive") it will
respond.

If you make changes to the plot and want it re-rendered you will need
to call `~.FigureCanvasBase.draw_idle` to request that the canvas be
re-drawn.  This method can be thought of *draw_soon* in analogy to
`asyncio.loop.call_soon`.

We can add this to our example above as ::

   def slow_loop(N, ln):
       for j in range(N):
           time.sleep(.1)  # to simulate some work
           if j % 10:
               ln.set_ydata(np.sin(((j // 10) % 5 * th)))
               ln.figure.canvas.draw_idle()

           ln.figure.canvas.flush_events()

   slow_loop(100, ln)


The more frequently you call `.FigureCanvasBase.flush_events` the more
responsive your figure will feel but at the cost of spending more
resources on the visualization and less on your computation.


.. _stale_artists:

Stale artists
=============

Artists (as of Matplotlib 1.5) have a **stale** attribute which is
`True` if the internal state of the artist has changed since the last
time it was rendered. By default the stale state is propagated up to
the Artists parents in the draw tree, e.g., if the color of a `.Line2D`
instance is changed, the `~.axes.Axes` and `.Figure` that
contain it will also be marked as "stale".  Thus, ``fig.stale`` will
report if any artist in the figure has been modified and is out of sync
with what is displayed on the screen.  This is intended to be used to
determine if ``draw_idle`` should be called to schedule a re-rendering
of the figure.

Each artist has a `.Artist.stale_callback` attribute which holds a callback
with the signature ::

  def callback(self: Artist, val: bool) -> None:
     ...

which by default is set to a function that forwards the stale state to
the artist's parent.   If you wish to suppress a given artist from propagating
set this attribute to None.

`.Figure` instances do not have a containing artist and their
default callback is `None`.  If you call `.pyplot.ion` and are not in
``IPython`` we will install a callback to invoke
`~.backend_bases.FigureCanvasBase.draw_idle` whenever the
`.Figure` becomes stale.  In ``IPython`` we use the
``'post_execute'`` hook to invoke
`~.backend_bases.FigureCanvasBase.draw_idle` on any stale figures
after having executed the user's input, but before returning the prompt
to the user.  If you are not using `.pyplot` you can use the callback
`Figure.stale_callback` attribute to be notified when a figure has
become stale.


.. _draw_idle:

Idle draw
=========

.. autosummary::
   :template: autosummary.rst
   :nosignatures:

   backend_bases.FigureCanvasBase.draw
   backend_bases.FigureCanvasBase.draw_idle
   backend_bases.FigureCanvasBase.flush_events


In almost all cases, we recommend using
`backend_bases.FigureCanvasBase.draw_idle` over
`backend_bases.FigureCanvasBase.draw`.  ``draw`` forces a rendering of
the figure whereas ``draw_idle`` schedules a rendering the next time
the GUI window is going to re-paint the screen.  This improves
performance by only rendering pixels that will be shown on the screen.  If
you want to be sure that the screen is updated as soon as possible do ::

  fig.canvas.draw_idle()
  fig.canvas.flush_events()



Threading
=========

Most GUI frameworks require that all updates to the screen, and hence
their main event loop, run on the main thread.  This makes pushing
periodic updates of a plot to a background thread impossible.
Although it seems backwards, it is typically easier to push your
computations to a background thread and periodically update
the figure on the main thread.

In general Matplotlib is not thread safe.  If you are going to update
`.Artist` objects in one thread and draw from another you should make
sure that you are locking in the critical sections.



Eventloop integration mechanism
===============================

CPython / readline
------------------

The Python C API provides a hook, :c:data:`PyOS_InputHook`, to register a
function to be run ("The function will be called when Python's
interpreter prompt is about to become idle and wait for user input
from the terminal.").  This hook can be used to integrate a second
event loop (the GUI event loop) with the python input prompt loop.
The hook functions typically exhaust all pending events on the GUI
event queue, run the main loop for a short fixed amount of time, or
run the event loop until a key is pressed on stdin.

Matplotlib does not currently do any management of :c:data:`PyOS_InputHook` due
to the wide range of ways that Matplotlib is used.  This management is left to
downstream libraries -- either user code or the shell.  Interactive figures,
even with Matplotlib in 'interactive mode', may not work in the vanilla python
repl if an appropriate :c:data:`PyOS_InputHook` is not registered.

Input hooks, and helpers to install them, are usually included with
the python bindings for GUI toolkits and may be registered on import.
IPython also ships input hook functions for all of the GUI frameworks
Matplotlib supports which can be installed via ``%matplotlib``.  This
is the recommended method of integrating Matplotlib and a prompt.


IPython / prompt_toolkit
------------------------

With IPython >= 5.0 IPython has changed from using CPython's readline
based prompt to a ``prompt_toolkit`` based prompt.  ``prompt_toolkit``
has the same conceptual input hook, which is fed into ``prompt_toolkit`` via the
:meth:`IPython.terminal.interactiveshell.TerminalInteractiveShell.inputhook`
method.  The source for the ``prompt_toolkit`` input hooks lives at
``IPython.terminal.pt_inputhooks``.



.. rubric:: Footnotes

.. [#f1] A limitation of this design is that you can only wait for one
     input, if there is a need to multiplex between multiple sources
     then the loop would look something like ::

       fds = [...]
           while True:                    # Loop
               inp = select(fds).read()   # Read
               eval(inp)                  # Evaluate / Print

.. [#f2] Or you can `write your own
         <https://www.youtube.com/watch?v=ZzfHjytDceU>`__ if you must.

.. [#f3] These examples are aggressively dropping many of the
     complexities that must be dealt with in the real world such as
     keyboard interrupts, timeouts, bad input, resource
     allocation and cleanup, etc.
