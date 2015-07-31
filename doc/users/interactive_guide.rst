.. _plotting-guide-interactive:

*******************
Interactive Figures
*******************

One of the most powerful uses of matplotlib is interactive
figures.  At the most basic matplotlib has the ability to zoom and pan
a figure to inspect your data, however there is also a full mouse and
keyboard event handling system to enable building sophisticated interactive
graphs.

This page is meant to be a rapid introduction to the relevant details of
integrating the matplotlib with a GUI event loop.


The GUI event loop
------------------

To handle asynchronous user input every GUI framework has an event
loop.  At the most basic this is a stack that can have events to be
processed.  In order for the GUI to be responsive this loop must be
run.  To manage this in python there are two basic methods:

1. let the GUI main loop block the python process
2. intermittently run the GUI loop for a period of time

Going with option 1 is going down the route of writing a bespoke GUI
application.  In this case, commonly refereed to as 'embedding', the
GUI event loop is running the show and you should not use the `pyplot`
layer.  Doing anything short of writing a full GUI application
requires option 2.

The python capi provides a hook, `PyOS_InputHook`, to register a
function to be run "The function will be called when Python's
interpreter prompt is about to become idle and wait for user input
from the terminal.".  This hook can be used to integrate a second
event loop with the python repl.  Such a hooks are usually included
with the python bindings for GUI toolkits and may be registered on
import.  IPython also includes hooks for all of the GUI frameworks
supported by matplotlib.  The hook functions typically exhaust
all pending events on the GUI event queue, run the main loop for a
short fixed amount of time, or run the event loop until a key is
pressed on stdin.

matplotlib does not currently do any management of `PyOS_InputHook`
due to the wide range of ways that matplotlib is used.  This
management is left to the code using matplotlib.  Interactive figures,
even with matplotlib in 'interactive mode', may not work in the
vanilla python repl if an appropriate `PyOS_InputHook` is not
registered.  We suggest using ``IPython``, which in addition to
improving the command line, ensures that such a `PyOS_InptuHook`
function is registered for you GUI backend of choice.

A drawback of relying on `PyOS_InputHook` is that the GUI event loop
is only processing events while python is otherwise idle and waiting
for user input.  If you want the GUI to be responsive during long
running code it is necessary to periodically flush the GUI event
queue.  To achive this, almost all of the of the GUI-based ``Canvas``
classes provide a `flush_event` method.  By periodically calling this
method the GUI will be updated and appear to be responsive.

In both cases, to schedule a re-draw of the figure at some point in
the future use ``fig.canvas.draw_idle()``.  This will defer the actual
rendering of the figure until the GUI is ready to update it's
on-screen representation.

Stale Artists
-------------

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
--------
