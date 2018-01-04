========================
 MEP22: Toolbar rewrite
========================

.. contents::
   :local:

Status
======
**Progress**


Branches and Pull requests
==========================

Previous work
 * https://github.com/matplotlib/matplotlib/pull/1849
 * https://github.com/matplotlib/matplotlib/pull/2557
 * https://github.com/matplotlib/matplotlib/pull/2465

Pull Requests:
 * Removing the NavigationToolbar classes
   https://github.com/matplotlib/matplotlib/pull/2740 **CLOSED**
 * Keeping the NavigationToolbar classes https://github.com/matplotlib/matplotlib/pull/2759 **CLOSED**
 * Navigation by events: https://github.com/matplotlib/matplotlib/pull/3652 **MERGED**
 * Implementation for QT https://github.com/matplotlib/matplotlib/pull/9934

Abstract
========

The main goal of this MEP is to make it easier to modify (add, change,
remove) the way the user interacts with the figures.

The user interaction with the figure is deeply integrated within the
Canvas and Toolbar. Making extremely difficult to do any modification.

This MEP proposes the separation of this interaction into Toolbar,
Navigation and Tools to provide independent access and
reconfiguration.

This approach will make easier to create and share tools among
users. In the far future, we can even foresee a kind of Marketplace
for `Tools` where the most popular can be added into the main
distribution.

Detailed description
====================

The reconfiguration of the Toolbar is complex, most of the time it
requires a custom backend.

The creation of custom Tools sometimes interferes with the Toolbar, as
example see https://github.com/matplotlib/matplotlib/issues/2694 also
the shortcuts are hardcoded and again not easily modifiable
https://github.com/matplotlib/matplotlib/issues/2699

The proposed solution is to take the actions out of the `Toolbar` and
the shortcuts out of the `Canvas`.  This actions and shortcuts will be
in the form of `Tools`.

A new class `Navigation` will be the bridge between the events from
the `Canvas` and `Toolbar` and redirect them to the appropiate `Tool`.

At the end the user interaction will be divided into three classes:

 * ToolManager: This class is instantiated for each FigureManager
   and connect the all user interactions with the Tools
 * ToolContainerBase: This class is relegated only as a GUI access
   to Tools.
 * ToolBase and ToolToggleBase: Are the basic definition of Tools.
 * StatusbarBase: This class is instatiated for each FigureManager, used to display messages


Implementation
==============

ToolBase(object)
----------------

Tools can have a graphical representation as the `SubplotTool` or not even be present in the Toolbar as `Quit`

The `ToolBase` has the following class attributes for configuration at definition time

 * default_keymap = None: Key(s) to be used to trigger the tool
 * description = '': Small description of the tool
 * image = None: Image that is used in the toolbar

The following instance attributes are set at instantiation:
 * name
 * toolmanager

**Methods**
 * trigger(self, sender, event, data=None): This is the main method of the Tool, it is called when the Tool is triggered by:
   * Toolbar button click
   * keypress associated with the Tool Keymap
   * Call to toolmanager.trigger_tool(name)
 * figure (property):
   * getter: returns the current figure that the tool is associated to
   * setter: Set the figure
 * ``destroy(self)``: Destroy the `Tool` graphical interface (if exists)

**Available Tools**
  * ToolHome
  * ToolBack
  * ToolForward
  * ToolZoom
  * ToolPan
  * ToolConfigureSubplots
  * ToolSaveFigure
  * ToolGrid
  * ToolMinorGrid
  * ToolFullScreen,
  * ToolQuit
  * ToolQuitAll
  * ToolEnableAllNavigation
  * ToolEnableNavigation
  * ToolXScale
  * ToolYScale
  * ToolCursorPosition
  * ToolViewsPositions
  * ToolSetCursor
  * ToolRubberband


ToolToggleBase(ToolBase)
------------------------

The `ToolToggleBase` has the following class attributes for
configuration at definition time

 * radio_group = None: Attribute to group 'radio' like tools (mutually
   exclusive)
 * cursor = None: Cursor to use when the tool is active
 * default_toggled = False: Initial state of the toggle

The **Toggleable** Tools, can capture keypress, mouse moves, and mouse
button press

It defines the following methods
 * enable(self, event): Called by `ToolToggleBase.trigger` method
 * disable(self, event): Called when the tool is untoggled
 * toggled : **Property** True or False

**Available Tools**
 * ToolZoom
 * ToolPan


ToolManager
-----------

Defines the following attributes
 * canvas:
 * keypresslock: Lock to know if the `canvas` key_press_event` is
        available and process it
 * messagelock: Lock to know if the message is available to write
 * figure (property):
   * getter: returns the current figure
   * setter: Set the figure

Public methods for **User use**:
 * toolmanager_connect(self, s, func): Connect to toolmanager for events
 * toolmanager_disconnect(self, cid): Disconnect from navigation event
 * message_event(self, message, sender=None): Emit a
   tool_message_event event
 * active_toggle(self): **Property** The currently toggled tools or
   None
 * get_tool_keymap(self, name): Return a list of keys that are
   associated with the tool
 * update_keymap(self, name, ``*keys``): Set the keys for the given tool
 * remove_tool(self, name): Removes tool from the navigation control.
 * add_tool(self, name, tool, *args, **kwargs): Add tool to ToolManager
 * trigger_tool(self, name, sender=None, canvasevent=None,
   data=None): Trigger a tool and fire the event
 * tools(self) **Property**: Return a dict with available tools with
   corresponding keymaps, descriptions and objects
 * get_tool(self, name): Return the tool object



ToolContainerBase
-----------------

Methods for **Backend implementation**
 * add_toolitem(self, name, group, position, image, description,
   toggle): Add a toolitem to the toolbar. This method is a callback
   from `tool_added_event` (emited by navigation)
 * toggle_toolitem(self, name): Toggle the toolitem without firing
   event.
 * remove_toolitem(self, name): Remove a toolitem from the `Toolbar`


StatusbarBase
-------------
Methods for **Backend implementation**
 * set_message(self, s): Display a message in the status bar



Backward compatibility
======================

For backward compatibility added a 'navigation' key to
`rcsetup.validate_toolbar`, that is used for Navigation classes
instantiation instead of the NavigationToolbar classes

With this parameter, it makes it transparent to anyone using the
existing backends.

[@pelson comment: This also gives us an opportunity to avoid needing
to implement all of this in the same PR - some backends can
potentially exist without the new functionality for a short while (but
it must be done at some point).]
