#------------------------------------------------------------------------------
# 
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#  
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#  
#  Author: David C. Morrill
#  Date:   09/01/2005
#
#------------------------------------------------------------------------------
""" Displays a message to the user as a modal window.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasPrivateTraits, Str, Any
    
from view \
    import View
    
from ui_traits \
    import buttons_trait
    
#-------------------------------------------------------------------------------
#  'Message' class:  
#-------------------------------------------------------------------------------

class Message ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    # The message to be displayed
    message = Str 

#-------------------------------------------------------------------------------
#  Displays a user specified message:  
#-------------------------------------------------------------------------------
        
def message ( message = '', title = 'Message', buttons = [ 'OK' ], 
              parent  = None ):
    """ Displays a message to the user as a model window with the specified
    title and buttons.
    
    If *buttons* is not specified, a single **OK** button is used, which is
    appropriate for notifications, where no further action or decision on the
    user's part is required.
    """
    msg = Message( message = message )
    ui  = msg.edit_traits( parent = parent,
                           view   = View( [ 'message~', '|<>' ],
                                          title   = title,
                                          buttons = buttons,
                                          kind    = 'modal' ) )
    return ui.result    
    
#-------------------------------------------------------------------------------
#  Displays a user specified error message:  
#-------------------------------------------------------------------------------
        
def error ( message = '', title = 'Message', buttons = [ 'OK', 'Cancel' ],
            parent  = None ):
    """ Displays a message to the user as a modal window with the specified
    title and buttons.
    
    If *buttons* is not specified, **OK** and **Cancel** buttons are used,
    which is appropriate for confirmations, where the user must decide whether
    to proceed. Be sure to word the message so that it is clear that clicking
    **OK** continues the operation.
    """
    msg = Message( message = message )
    ui  = msg.edit_traits( parent = parent,
                           view   = View( [ 'message~', '|<>' ],
                                          title   = title,
                                          buttons = buttons,
                                          kind    = 'modal' ) )
    return ui.result
    
