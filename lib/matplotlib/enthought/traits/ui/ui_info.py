#-------------------------------------------------------------------------------
#
#  Define the UIInfo class used to represent the object and editor content of
#  an active traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/13/2004
#
#  Symbols defined: Info
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import HasPrivateTraits, ReadOnly, Constant

#-------------------------------------------------------------------------------
#  'UIInfo' class:
#-------------------------------------------------------------------------------

class UIInfo ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    ui = ReadOnly  # Bound to a UI object at UIInfo construction time
    
    #---------------------------------------------------------------------------
    #  Bind's all of the associated context objects as traits of the object:   
    #---------------------------------------------------------------------------
        
    def bind_context ( self ):
        """ Bind's all of the associated context objects as traits of the 
            object.
        """
        for name, value in self.ui.context.items():
            self.bind( name, value )
                
    #---------------------------------------------------------------------------
    #  Binds a name to a value if it is not already bound:
    #---------------------------------------------------------------------------
                
    def bind ( self, name, value ):
        """ Binds a name to a value if it is not already bound.
        """
        if not hasattr( self, name ):
            self.add_trait( name, Constant( value ) )
            self.ui._names.append( name )
        
