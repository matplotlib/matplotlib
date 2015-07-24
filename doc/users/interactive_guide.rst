.. _plotting-guide-interactive:

*******************
Interactive Figures
*******************

One of the most powerful ways to use matplotlib is interactive
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
from the terminal.".  As an implementation detail of cpython when
using readline this times out every 0.1 seconds.  Using this hook a
second event loop can be integrated with the terminal.  This is done
in the cpython source for tk, by some GUI framework code (such as
pyqt), and by IPython.  These function typically either exhaust all
pending events on the GUI event queue or run the main loop for a short
fixed amount of time.  matplotlib does not currently do any management
of `PyOS_InputHook` due to the wide range of use-cases, this
management is left to the code using matplotlib.  Due to this,
interactive figures, even with matplotlib in 'interactive mode' may
not be reliable in the vanilla python repl, we suggest using IPython
which ensures that such a function is registered for you GUI backend
of choice.

A draw back of the above approach is that in is only useful while
python is otherwise idle and waiting for user input.  The exact
methods required to force the GUI to process it's event loop varies
between frameworks.  To enable writing GUI agnostic code, almost all
of the GUI-based ``Canvas`` classes provide a `flush_event` method.
By periodically calling this method the GUI can appear to be
updated and appear to be responsive.

In both cases, scheduling a re-draw of the figure at some point in the
future use ``fig.canvas.draw_idle()``.  This will defer the actual
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
