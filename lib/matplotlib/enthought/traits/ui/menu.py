#-------------------------------------------------------------------------------
#  
#  Defines the standard menu bar for use with Traits UI windows and panels.
#  
#  Written by: David C. Morrill
#  
#  Date: 12/19/2004
#  
#  (c) Copyright 2004 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

# Import and rename the needed PyFace elements:
from matplotlib.enthought.pyface.action import MenuBarManager as MenuBar
from matplotlib.enthought.pyface.action import MenuManager    as Menu
from matplotlib.enthought.pyface.action import Group          as ActionGroup
from matplotlib.enthought.pyface.action import Action

#-------------------------------------------------------------------------------
#  Standard actions and menu bar definitions:
#-------------------------------------------------------------------------------

# Menu separator:
Separator = ActionGroup

# The standard 'close window' action:
CloseAction = Action(
    name   = 'Close',
    action = '_on_close'
)

# The standard 'undo last change' action:
UndoAction = Action(
    name         = 'Undo',
    action       = '_on_undo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard 'redo last undo' action:
RedoAction = Action(
    name         = 'Redo',
    action       = '_on_redo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_redo'
)

# The standard 'Revert all changes' action:
RevertAction = Action(
    name         = 'Revert',
    action       = '_on_revert',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard 'Show help' action:
HelpAction = Action(
    name   = 'Help',
    action = 'show_help'
)

# The standard Trait's UI menu bar:
StandardMenuBar = MenuBar(
    Menu( CloseAction,
          name = 'File' ),
    Menu( UndoAction,
          RedoAction,
          RevertAction,
          name = 'Edit' ),
    Menu( HelpAction,
          name = 'Help' )
)

