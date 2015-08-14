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
