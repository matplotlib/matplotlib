#-------------------------------------------------------------------------------
#
#  Define the help interface for displaying the help associated with a traits
#  UI View.
#
#  Written by: David C. Morrill
#
#  Date: 02/04/2005
#
#  Symbols defined: 
#
#  (c) Copyright 2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from toolkit import toolkit
        
#-------------------------------------------------------------------------------
#  Default handler for showing the help associated with a view:  
#-------------------------------------------------------------------------------
            
def default_show_help ( info, control ):
    """ Default handler for showing the help associated with a view.
    """
    toolkit().show_help( info.ui, control )
    
# Set up the default show help handler:    
show_help = default_show_help    

#-------------------------------------------------------------------------------
#  Allows an application to change the default show help handler:  
#-------------------------------------------------------------------------------

def on_help_call ( self, new_show_help = None ):
    global show_help
    
    result = show_help
    if new_show_help is not None:
        show_help = new_show_help
    return result
    
