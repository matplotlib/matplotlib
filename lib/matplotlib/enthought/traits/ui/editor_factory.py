#-------------------------------------------------------------------------------
#
#  Define the abstract EditorFactory class used to represent a factory for 
#  creating the Editor objects used in a traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: EditorFactory
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import HasPrivateTraits

#-------------------------------------------------------------------------------
#  'EditorFactory' abstract base class:
#-------------------------------------------------------------------------------

class EditorFactory ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, *args, **traits ):
        """ Initializes the object.
        """
        HasPrivateTraits.__init__( self, **traits )
        self.init( *args )
        
    #---------------------------------------------------------------------------
    #  Performs any initialization needed after all constructor traits have 
    #  been set:
    #---------------------------------------------------------------------------
     
    def init ( self ):
        """ Performs any initialization needed after all constructor traits 
            have been set.
        """
        pass
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def custom_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def text_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def readonly_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError

