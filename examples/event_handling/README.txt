.. _event_handling_examples:

Event handling
==============

Matplotlib supports :doc:`event handling</users/explain/event_handling>` with 
a GUI neutral event model, so you can connect to Matplotlib events without 
knowledge of what user interface Matplotlib will ultimately be plugged in to.  
This has two advantages: the code you write will be more portable, and 
Matplotlib events are aware of things like data coordinate space and which 
axes the event occurs in so you don't have to mess with low level 
transformation details to go from canvas space to data space.  Object picking 
examples are also included.
