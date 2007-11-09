#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: David C. Morrill
# Date: 12/19/2004
#------------------------------------------------------------------------------
""" Defines the standard menu bar for use with Traits UI windows and panels, 
and standard actions and buttons.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Str

# Import and rename the needed PyFace elements:
from enthought.pyface.action.api \
    import ToolBarManager as ToolBar
    
from enthought.pyface.action.api \
    import MenuBarManager as MenuBar
    
from enthought.pyface.action.api \
    import MenuManager as Menu
    
from enthought.pyface.action.api \
    import Group as ActionGroup
    
from enthought.pyface.action.api \
    import Action as PyFaceAction

#-------------------------------------------------------------------------------
#  'Action' class (extends the core pyface Action class):  
#-------------------------------------------------------------------------------

class Action ( PyFaceAction ):
    """ An action on a menu bar in a Traits UI window or panel.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # Pre-condition for showing the action. If the expression evaluates to False,
    # the action is not visible (and disappears if it was previously visible).
    # If the value evaluates to True, the action becomes visible. All
    # **visible_when** conditions are checked each time that any trait value
    # is edited in the display. Therefore, you can use **visible_when**
    # conditions to hide or show actions in response to user input.
    visible_when = Str
        
    # Pre-condition for enabling the action. If the expression evaluates to 
    # False, the action is disabled, that is, it cannot be selected. All
    # **enabled_when** conditions are checked each time that any trait value
    # is edited in the display. Therefore, you can use **enabled_when** 
    # conditions to enable or disable actions in response to user input.
    enabled_when = Str
    
    # Boolean expression indicating when the action is displayed with a check
    # mark beside it. This attribute applies only to actions that are included
    # in menus.
    checked_when = Str
    
    # Pre-condition for including the action in the menu bar or toolbar. If the
    # expression evaluates to False, the action is not defined in the display.
    # Conditions for **defined_when** are evaluated only once, when the display
    # is first constructed. 
    defined_when = Str

    # The method to call to perform the action, on the Handler for the window.
    # The method must accept a single parameter, which is a UIInfo object.
    # Because Actions are associated with Views rather than Handlers, you must
    # ensure that the Handler object for a particular window has a method with
    # the correct name, for each Action defined on the View for that window.
    action = Str

#-------------------------------------------------------------------------------
#  Standard actions and menu bar definitions:
#-------------------------------------------------------------------------------

# Menu separator
Separator = ActionGroup

# The standard "close window" action
CloseAction = Action(
    name   = 'Close',
    action = '_on_close'
)

# The standard "undo last change" action
UndoAction = Action(
    name         = 'Undo',
    action       = '_on_undo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard "redo last undo" action
RedoAction = Action(
    name         = 'Redo',
    action       = '_on_redo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_redo'
)

# The standard "revert all changes" action
RevertAction = Action(
    name         = 'Revert',
    action       = '_on_revert',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard "show help" action
HelpAction = Action(
    name   = 'Help',
    action = 'show_help'
)

# The standard Traits UI menu bar
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

#-------------------------------------------------------------------------------
#  Standard buttons (i.e. actions):  
#-------------------------------------------------------------------------------

NoButton     = Action( name = '' )

# Appears as two buttons: **Undo** and **Redo**. When **Undo** is clicked, the
# most recent change to the data is cancelled, restoring the previous value.
# **Redo** cancels the most recent "undo" operation.
UndoButton   = Action( name = 'Undo' )

# When the user clicks the **Revert** button, all changes made in the window are
# cancelled and the original values are restored. If the changes have been 
# applied to the model (because the user clicked **Apply** or because the window
# is live), the model data is restored as well. The window remains open.
RevertButton = Action( name = 'Revert' )

# When theuser clicks the **Apply** button, all changes made in the window are 
# applied to the model. This option is meaningful only for modal windows.
ApplyButton  = Action( name = 'Apply' )

# When the user clicks the **OK** button, all changes made in the window are
# applied to the model, and the window is closed.
OKButton     = Action( name = 'OK' )

# When the user clicks the **Cancel** button, all changes made in the window
# are discarded; if the window is live, the model is restored to the values it
# held before the window was opened. The window is then closed.
CancelButton = Action( name = 'Cancel' )

# When the user clicks the **Help** button, the current help handler is 
# invoked. If the default help handler is used, a pop-up window is displayed,
# which contains the **help** text for the top-level Group (if any), and for 
# the items in the view. If the default help handler has been overridden,
# the action is determined by the custom help handler. See
# **enthought.traits.ui.help**.
HelpButton   = Action( name = 'Help' )

OKCancelButtons = [ OKButton, CancelButton ]
ModalButtons = [ ApplyButton, RevertButton, OKButton, CancelButton, HelpButton ]
LiveButtons  = [ UndoButton,  RevertButton, OKButton, CancelButton, HelpButton ]
# The window has no command buttons.
NoButtons    = [ NoButton ]


