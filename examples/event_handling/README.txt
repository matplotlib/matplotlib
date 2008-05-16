matplotlib event handling
=========================

matplotlib supports event handling with a GUI neutral event model.  So
you can connect to matplotlib events w/o knowledge of what user
interface matplotlib will ultimately be plugged in to.  This has two
advantages: the code you write will be more portable, and matplotlib
events are aware of things like data coordinate space and whih axes
the event occurs in so you don't have to mess with low level
transformation details to go from canvas space to data space.  Object
picking examples are also included.

There is an event handling tutorial at
http://matplotlib.sourceforge.net/pycon/event_handling_tut.pdf.  The
ReST source for this document is included in the matplotlib source
distribution.


