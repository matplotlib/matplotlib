======================================
 MEP27: decouple pyplot from backends
======================================

.. contents::
   :local:


Status
======
**Discussion**

Branches and Pull requests
==========================
Main PR (including GTK3):
+ https://github.com/matplotlib/matplotlib/pull/4143

Backend specific branch diffs:
+ https://github.com/OceanWolf/matplotlib/compare/backend-refactor...OceanWolf:backend-refactor-tkagg
+ https://github.com/OceanWolf/matplotlib/compare/backend-refactor...OceanWolf:backend-refactor-qt
+ https://github.com/OceanWolf/matplotlib/compare/backend-refactor...backend-refactor-wx

Abstract
========

This MEP refactors the backends to give a more structured and
consistent API, removing generic code and consolidate existing code.
To do this we propose splitting:

1. ``FigureManagerBase`` and its derived classes into the core
   functionality class ``FigureManager`` and a backend specific class
   ``WindowBase`` and
2. ``ShowBase`` and its derived classes into ``Gcf.show_all`` and ``MainLoopBase``.

Detailed description
====================

This MEP aims to consolidate the backends API into one single uniform
API, removing generic code out of the backend (which includes
``_pylab_helpers`` and ``Gcf``), and push code to a more appropriate
level in matplotlib.  With this we automatically remove
inconsistencies that appear in the backends, such as
``FigureManagerBase.resize(w, h)`` which sometimes sets the canvas,
and other times set the entire window to the dimensions given,
depending on the backend.

Two main places for generic code appear in the classes derived from
``FigureManagerBase`` and ``ShowBase``.

1. ``FigureManagerBase`` has **three** jobs at the moment:

    1. The documentation describes it as a *``Helper class for pyplot
       mode, wraps everything up into a neat bundle''*
    2. But it doesn't just wrap the canvas and toolbar, it also does
       all of the windowing tasks itself.  The conflation of these two
       tasks gets seen the best in the following line: ```python
       self.set_window_title("Figure %d" % num) ``` This combines
       backend specific code ``self.set_window_title(title)`` with
       matplotlib generic code ``title = "Figure %d" % num``.

    3. Currently the backend specific subclass of ``FigureManager``
       decides when to end the mainloop.  This also seems very wrong
       as the figure should have no control over the other figures.


2. ``ShowBase`` has two jobs:

    1. It has the job of going through all figure managers registered
       in ``_pylab_helpers.Gcf`` and telling them to show themselves.
    2. And secondly it has the job of performing the backend specific
       ``mainloop`` to block the main programme and thus keep the
       figures from dying.

Implementation
==============

The description of this MEP gives us most of the solution:

1. To remove the windowing aspect out of ``FigureManagerBase`` letting
   it simply wrap this new class along with the other backend classes.
   Create a new ``WindowBase`` class that can handle this
   functionality, with pass-through methods (:arrow_right:) to
   ``WindowBase``.  Classes that subclass ``WindowBase`` should also
   subclass the GUI specific window class to ensure backward
   compatibility (``manager.window == manager.window``).
2. Refactor the mainloop of ``ShowBase`` into ``MainLoopBase``, which
   encapsulates the end of the loop as well.  We give an instance of
   ``MainLoop`` to ``FigureManager`` as a key unlock the exit method
   (requiring all keys returned before the loop can die).  Note this
   opens the possibility for multiple backends to run concurrently.
3. Now that ``FigureManagerBase`` has no backend specifics in it, to
   rename it to ``FigureManager``, and move to a new file
   ``backend_managers.py`` noting that:

   1. This allows us to break up the conversion of backends into
      separate PRs as we can keep the existing ``FigureManagerBase``
      class and its dependencies intact.
   2. and this also anticipates MEP22 where the new
      ``NavigationBase`` has morphed into a backend independent
      ``ToolManager``.

+--------------------------------------+------------------------------+---------------------+--------------------------------+
|FigureManagerBase(canvas, num)        |FigureManager(figure, num)    |``WindowBase(title)``|Notes                           |
|                                      |                              |                     |                                |
+======================================+==============================+=====================+================================+
|show                                  |                              |show                 |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|destroy                               |calls destroy on all          |destroy              |                                |
|                                      |components                    |                     |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|full_screen_toggle                    |handles logic                 |set_fullscreen       |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|resize                                |                              |resize               |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|key_press                             |key_press                     |                     |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|show_popup                            |show_poup                     |                     |Not used anywhere in mpl, and   |
|                                      |                              |                     |does nothing.                   |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|get_window_title                      |                              |get_window_title     |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|set_window_title                      |                              |set_window_title     |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|                                      |_get_toolbar                  |                     |A common method to all          |
|                                      |                              |                     |subclasses of FigureManagerBase |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|                                      |                              |set_default_size     |                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+
|                                      |                              |add_element_to_window|                                |
+--------------------------------------+------------------------------+---------------------+--------------------------------+


+----------+------------+-------------+
|ShowBase  |MainLoopBase|Notes        |
+==========+============+=============+
|mainloop  |begin       |             |
+----------+------------+-------------+
|          |end         |Gets called  |
|          |            |automagically|
|          |            |when no more |
|          |            |instances of |
|          |            |the subclass |
|          |            |exist        |
+----------+------------+-------------+
|__call__  |            |Method moved |
|          |            |to           |
|          |            |Gcf.show_all |
+----------+------------+-------------+

Future compatibility
====================

As eluded to above when discussing MEP 22, this refactor makes it easy
to add in new generic features.  At the moment, MEP 22 has to make
ugly hacks to each class extending from ``FigureManagerBase``.  With
this code, this only needs to get made in the single ``FigureManager``
class.  This also makes the later deprecation of
``NavigationToolbar2`` very straightforward, only needing to touch the
single ``FigureManager`` class

MEP 23 makes for another use case where this refactored code will come
in very handy.

Backward compatibility
======================

As we leave all backend code intact, only adding missing methods to
existing classes, this should work seamlessly for all use cases.  The
only difference will lie for backends that used
``FigureManager.resize`` to resize the canvas and not the window, due
to the standardisation of the API.

I would envision that the classes made obsolete by this refactor get
deprecated and removed on the same timetable as
``NavigationToolbar2``, also note that the change in call signature to
the ``FigureCanvasWx`` constructor, while backward compatible, I think
the old (imho ugly style) signature should get deprecated and removed
in the same manner as everything else.

+-------------------------+-------------------------+-------------------------+
|backend                  |manager.resize(w,h)      |Extra                    |
+=========================+=========================+=========================+
|gtk3                     |window                   |                         |
+-------------------------+-------------------------+-------------------------+
|Tk                       |canvas                   |                         |
+-------------------------+-------------------------+-------------------------+
|Qt                       |window                   |                         |
+-------------------------+-------------------------+-------------------------+
|Wx                       |canvas                   |FigureManagerWx had      |
|                         |                         |``frame`` as an alias to |
|                         |                         |window, so this also     |
|                         |                         |breaks BC.               |
+-------------------------+-------------------------+-------------------------+


Alternatives
============

If there were any alternative solutions to solving the same problem,
they should be discussed here, along with a justification for the
chosen approach.

Questions
=========

Mdehoon: Can you elaborate on how to run multiple backends
concurrently?

OceanWolf: @mdehoon, as I say, not for this MEP, but I see this MEP
opens it up as a future possibility.  Basically the ``MainLoopBase``
class acts a per backend Gcf, in this MEP it tracks the number of
figures open per backend, and manages the mainloops for those
backends.  It closes the backend specific mainloop when it detects
that no figures remain open for that backend.  Because of this I
imagine that with only a small amount of tweaking that we can do
full-multi-backend matplotlib.  No idea yet why one would want to, but
I leave the possibility there in MainLoopBase.  With all the
backend-code specifics refactored out of ``FigureManager`` also aids
in this, one manager to rule them (the backends) all.

Mdehoon: @OceanWolf, OK, thanks for the explanation. Having a uniform
API for the backends is very important for the maintainability of
matplotlib. I think this MEP is a step in the right direction.
