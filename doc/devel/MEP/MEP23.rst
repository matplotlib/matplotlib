========================================
 MEP23: Multiple Figures per GUI window
========================================

.. contents::
   :local:



Status
======

**Discussion**

Branches and Pull requests
==========================

**Previous work**
- https://github.com/matplotlib/matplotlib/pull/2465 **To-delete**


Abstract
========

Add the possibility to have multiple figures grouped under the same
`FigureManager`

Detailed description
====================

Under the current structure, every canvas has its own window.

This is and may continue to be the desired method of operation for
most use cases.

Sometimes when there are too many figures open at the same time, it is
desirable to be able to group these under the same window
[see](https://github.com/matplotlib/matplotlib/issues/2194).

The proposed solution modifies `FigureManagerBase` to contain and manage more
than one `canvas`.  The settings parameter :rc:`backend.multifigure` control
when the **MultiFigure** behaviour is desired.

**Note**

It is important to note, that the proposed solution, assumes that the
[MEP22](https://github.com/matplotlib/matplotlib/wiki/Mep22) is
already in place. This is simply because the actual implementation of
the `Toolbar` makes it pretty hard to switch between canvases.

Implementation
==============

The first implementation will be done in `GTK3` using a Notebook as
canvas container.

`FigureManagerBase`
-------------------

will add the following new methods

* `add_canvas`: To add a canvas to an existing `FigureManager` object
* `remove_canvas`: To remove a canvas from a `FigureManager` object,
  if it is the last one, it will be destroyed
* `move_canvas`: To move a canvas from one `FigureManager` to another.
* `set_canvas_title`: To change the title associated with a specific
  canvas container
* `get_canvas_title`: To get the title associated with a specific
  canvas container
* `get_active_canvas`: To get the canvas that is in the foreground and
  is subject to the gui events. There is no `set_active_canvas`
  because the active canvas, is defined when `show` is called on a
  `Canvas` object.

`new_figure_manager`
--------------------

To control which `FigureManager` will contain the new figures, an
extra optional parameter `figuremanager` will be added, this parameter
value will be passed to `new_figure_manager_given_figure`

`new_figure_manager_given_figure`
---------------------------------

* If `figuremanager` parameter is give, this `FigureManager` object
  will be used instead of creating a new one.
* If `rcParams['backend.multifigure'] == True`: The last
  `FigureManager` object will be used instead of creating a new one.

`NavigationBase`
----------------

Modifies the `NavigationBase` to keep a list of canvases, directing
the actions to the active one

Backward compatibility
======================

For the **MultiFigure** properties to be visible, the user has to
activate them directly setting `rcParams['backend.multifigure'] =
True`

It should be backwards compatible for backends that adhere to the
current `FigureManagerBase` structure even if they have not
implemented the **MultiFigure** magic yet.


Alternatives
============

Insted of modifying the `FigureManagerBase` it could be possible to add
a parallel class, that handles the cases where
`rcParams['backend.multifigure'] = True`.  This will warranty that
there won't be any problems with custom made backends, but also makes
bigger the code, and more things to maintain.
