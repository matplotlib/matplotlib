#-------------------------------------------------------------------------------
#
#  Define the Handler class used to manage and control the editing process in a
#  traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: Handler
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from toolkit          import toolkit
from help             import show_help

from matplotlib.enthought.traits import HasPrivateTraits

#-------------------------------------------------------------------------------
#  'Handler' class:
#-------------------------------------------------------------------------------

class Handler ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Initializes the controls of a user interface:
    #---------------------------------------------------------------------------
    
    def init ( self, info ):
        """ Initializes the controls of a user interface.
        """
        return True
        
    #---------------------------------------------------------------------------
    #  Positions a dialog-based user interface on the display:
    #---------------------------------------------------------------------------
        
    def position ( self, info ):
        """ Positions a dialog-based user interface on the display.
        """
        toolkit().position( info.ui )
        
    #---------------------------------------------------------------------------
    #  Handles a request to close a dialog-based user interface by the user:
    #---------------------------------------------------------------------------
        
    def close ( self, info, is_ok ):
        """ Handles a request to close a dialog-based user interface by the 
            user.
        """
        return True
        
    #---------------------------------------------------------------------------
    #  Handles a dialog-based user interface being closed by the user:
    #---------------------------------------------------------------------------
        
    def closed ( self, info, is_ok ):
        """ Handles a dialog-based user interface being closed by the user.
        """
        return
        
    #---------------------------------------------------------------------------
    #  Shows the help associated with the view:  
    #---------------------------------------------------------------------------
                
    def show_help ( self, info, control = None ):
        """ Shows the help associated with the view.
        """
        if control is None:
            control = info.ui.control
        show_help( info, control )
        
    #---------------------------------------------------------------------------
    #  Handles setting a specified object trait's value:
    #---------------------------------------------------------------------------
        
    def setattr ( self, object, name, value ):
        """ Handles setting a specified object trait's value.
        """
        setattr( object, name, value )
        
    #---------------------------------------------------------------------------
    #  Edits the object's traits: (Overrides HasTraits)
    #---------------------------------------------------------------------------
    
    def edit_traits ( self, view    = None, parent = None, kind = None, 
                            context = None ): 
        """ Edits the object's traits.
        """
        if context is None:
            context = self
        return self.trait_view( view ).ui( context, parent, kind, 
                                           self.trait_view_elements(), self )
        
    #---------------------------------------------------------------------------
    #  Configure the object's traits:
    #---------------------------------------------------------------------------
    
    def configure_traits ( self, filename = None, view    = None, 
                                 kind     = None, edit    = True, 
                                 context  = None, handler = None ):
        super( HasPrivateTraits, self ).configure_traits(
                          filename, view, kind, edit, context, handler or self )
   
    #---------------------------------------------------------------------------
    #  Handles an 'Undo' change request:
    #---------------------------------------------------------------------------
           
    def _on_undo ( self, info ):
        """ Handles an 'Undo' change request.
        """
        if info.ui.history is not None:
            info.ui.history.undo()
   
    #---------------------------------------------------------------------------
    #  Handles a 'Redo' change request:
    #---------------------------------------------------------------------------
           
    def _on_redo ( self, info ):
        """ Handles a 'Redo' change request.
        """
        if info.ui.history is not None:
            info.ui.history.redo()
   
    #---------------------------------------------------------------------------
    #  Handles a 'Revert' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_revert ( self, info ):
        """ Handles a 'Revert' all changes request.
        """
        if info.ui.history is not None:
            info.ui.history.revert()
    
    #---------------------------------------------------------------------------
    #  Handles a 'Close' request:
    #---------------------------------------------------------------------------
           
    def _on_close ( self, info ):
        """ Handles a 'Close' request.
        """
        if (info.ui.owner is not None) and self.close( info, True ):
            info.ui.owner.close()
        
#-------------------------------------------------------------------------------
#  Default handler:  
#-------------------------------------------------------------------------------
                
_default_handler = Handler()

def default_handler ( handler = None ):
    global _default_handler
    
    if isinstance( handler, Handler ):
        _default_handler = handler
    return _default_handler
