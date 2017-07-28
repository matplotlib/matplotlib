.. _plotting-guide-interactive:

************************************************
Interactive Figures and Asynchronous Programming
************************************************

One of the most powerful uses of matplotlib is interactive
figures.  At the most basic matplotlib has the ability to zoom and pan
a figure to inspect your data, however there is also a full mouse and
keyboard event handling system to enable building sophisticated interactive
graphs.

This page is meant to be a rapid introduction to the relevant details
of integrating the matplotlib with a GUI event loop.  For further
details see `Interactive Applications using Matplotlib
<http://www.amazon.com/Interactive-Applications-using-Matplotlib-Benjamin/dp/1783988843>`__.

Fundamentally, all user interaction (and networking) is implemented as
an infinite loop waiting for events from the OS and then doing
something about it.  For example, a minimal Read Evaluate Print Loop
(REPL) is ::

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
interpreting the input and then doing something about it.

In practice most users do not work directly with these loops, and
instead framework that provides a mechanism to register callbacks
[#2]_.  This allows users to write reactive, event-driven, programs
without having to delve into the nity-grity [#f3]_ details of I/O.
Examples include ``observe`` and friends in `traitlets`, the
``Signal`` / ``Slot`` framework in Qt and the analogs in Gtk / Tk /
Wx, the request functions in web frameworks, or Matplotlib's
native :ref:`event handling system <event-handling-tutorial>`.



references to trackdown:
 - link to cpython REPL loop
 - curio / trio / asycio / twisted / tornado event loops
 - Beazly talk or two on asyncio


GUI to Matplotlib Bridge
------------------------

When using


Command Prompt
--------------

To handle asynchronous user input every GUI framework has an event
loop.  At the most basic this is a stack that can have events to be
processed.  In order for the GUI to be responsive this loop must be
run.  To manage this in python there are two basic methods:

1. let the GUI main loop block the python process
2. intermittently run the GUI loop for a period of time


Blocking
********


Interactive
***********

Scripts
-------

 - if you want always reactive figures while the script runs, you have to
   call `flush_event`
 - if you want to have reactive figures that block the script until they are closed (ex for
   collecting user input before continuing use


Full embedding
--------------

 - just let the underlying GUI event loop handle eve

Web
---

The Weeds
=========


The GUI event loop
------------------


The python capi provides a hook, `PyOS_InputHook`, to register a
function to be run "The function will be called when Python's
interpreter prompt is about to become idle and wait for user input
from the terminal.".  This hook can be used to integrate a second
event loop (the GUI event loop) with the python input prompt loop.
Such hooks are usually included with the python bindings for GUI
toolkits and may be registered on import.  IPython also includes hooks
for all of the GUI frameworks supported by matplotlib.  The hook
functions typically exhaust all pending events on the GUI event queue,
run the main loop for a short fixed amount of time, or run the event
loop until a key is pressed on stdin.

matplotlib does not currently do any management of `PyOS_InputHook`
due to the wide range of ways that matplotlib is used.  This
management is left to the code using matplotlib.  Interactive figures,
even with matplotlib in 'interactive mode', may not work in the
vanilla python repl if an appropriate `PyOS_InputHook` is not
registered.  We suggest using ``IPython``, which in addition to
improving the command line, ensures that such a `PyOS_InputHook`
function is registered for you GUI backend of choice.

A drawback of relying on `PyOS_InputHook` is that the GUI event loop
is only processing events while python is otherwise idle and waiting
for user input.  If you want the GUI to be responsive during long
running code it is necessary to periodically flush the GUI event
queue.  To achieve this, almost all of the of the GUI-based ``Canvas``
classes provide a `flush_event` method.  By periodically calling this
method the GUI will be updated and appear to be responsive.

In both cases, to schedule a re-draw of the figure at some point in
the future use ``fig.canvas.draw_idle()``.  This will defer the actual
rendering of the figure until the GUI is ready to update its
on-screen representation.

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


Interactive Mode
----------------


Blitting
========


.. ruberic:: Fotenotes

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
