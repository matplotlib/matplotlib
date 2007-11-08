#-------------------------------------------------------------------------------
#  
#  Written by: David C. Morrill
#  
#  Date: 12/14/2005
#  
#  (c) Copyright 2005 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------
""" Defines the DockableViewElement class, which allows Traits UIs and 
Traits UI elements to be docked in external PyFace DockWindow windows.
"""
#-------------------------------------------------------------------------------
#  Imports:  
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasPrivateTraits, Instance, false
    
from ui \
    import UI
    
from group \
    import Group
    
from view \
    import View
    
from view_element \
    import ViewSubElement
    
from enthought.pyface.dock.idockable \
    import IDockable
    
#-------------------------------------------------------------------------------
#  'DockableViewElement' class:  
#-------------------------------------------------------------------------------

class DockableViewElement ( HasPrivateTraits, IDockable ):
    """ Allows Traits UIs and Traits UI elements to be docked in external
    PyFace DockWindow windows.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # The Traits UI that can be docked with an external DockWindow
    ui = Instance( UI )
    
    # The (optional) element of the Traits UI that can be docked
    element = Instance( ViewSubElement )
    
    # Should the DockControl be closed on redocking?
    should_close = false
    
#-- IDockable interface --------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #  Should the current DockControl be closed before creating the new one:  
    #---------------------------------------------------------------------------
        
    def dockable_should_close ( self ):
        """ Should the current DockControl be closed before creating the new 
            one?
        """
        element = self.element
        if element is None:
            element = self.ui.view.content 
                
        if not isinstance( element, Group ):
            element = Group().set( content = [ element ] )
            
        group      = Group().set( content = [ element ] )
        self._view = View().set( **self.ui.view.get() ).set( content = group,
                                                             title   = '' )
                                                        
        return (self.should_close or (self.element is None))

    #---------------------------------------------------------------------------
    #  Gets a control that can be docked into a DockWindow:  
    #---------------------------------------------------------------------------
    
    def dockable_get_control ( self, parent ):
        """ Gets a control that can be docked into a DockWindow.
        """
        # Create the new UI:  
        ui = self._view.ui( self.ui.context, parent  = parent,
                                             kind    = 'subpanel', 
                                             handler = self.ui.handler )
                                             
        # Discard the reference to the view created previously:                                             
        self._view = None

        # If the old UI was closed, then switch to using the new one:                                             
        if self.element is None:
            self.ui = ui
        else:
            self._ui = ui
            
        return ui.control
        
    #---------------------------------------------------------------------------
    #  Allows the object to override the default DockControl settings:  
    #---------------------------------------------------------------------------
                
    def dockable_init_dockcontrol ( self, dock_control ):
        """ Allows the object to override the default DockControl settings.
        """
        dockable = self
        if self.element is not None:
            dockable = DockableViewElement( ui           = self._ui,
                                            element      = self.element,
                                            should_close = True )
            self._ui = None
            
        dock_control.set( dockable = dockable,
                          on_close = dockable.close_dock_control )

    #---------------------------------------------------------------------------
    #  Handles the closing of a DockControl containing a Traits UI:  
    #---------------------------------------------------------------------------
                        
    def close_dock_control ( self, dock_control, abort ):
        """ Handles the closing of a DockControl containing a Traits UI.
        """
        ui = self.ui
    
        # Ask the traits UI handler if it is OK to close the window:
        if (not abort) and (not ui.handler.close( ui.info, True )):
            # If not, tell the DockWindow not to close it:
            return False
    
        # Otherwise, clean up and close the traits UI:
        ui.dispose( abort = abort )

        # And tell the DockWindow to remove the DockControl:
        return True

