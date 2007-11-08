#-------------------------------------------------------------------------------
#    
#  CheckListEditor test case for Traits UI 
#    
#  Written by: David C. Morrill
#    
#  Date: 06/29/2005
#    
#  (c) Copyright 2005 by Enthought, Inc.  
#    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:  
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Enum, List, Str
    
from enthought.traits.ui.api \
    import Handler, View, Item, CheckListEditor
            
#-------------------------------------------------------------------------------
#  'CheckListTest' class:  
#-------------------------------------------------------------------------------
                        
class CheckListTest ( Handler ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    value       = List( editor = CheckListEditor( name = 'values', cols = 5 ) )
    values      = List( Str )
    values_text = Str( 'red orange yellow green blue indigo violet' )
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------
        
    simple_view = View( 'value',  'values_text@' )
    custom_view = View( 'value@', 'values_text@' )
                        
    #---------------------------------------------------------------------------
    #  'Initializes the object:  
    #---------------------------------------------------------------------------
                               
    def __init__ ( self, **traits ):
        super( CheckListTest, self ).__init__( **traits )
        self._values_text_changed()
    
    #---------------------------------------------------------------------------
    #  Event handlers:  
    #---------------------------------------------------------------------------
        
    def _values_text_changed ( self ):
        self.values = self.values_text.split()
            
#-------------------------------------------------------------------------------
#  Run the tests:  
#-------------------------------------------------------------------------------
                 
if __name__ == '__main__':
    clt = CheckListTest()
    clt.configure_traits( view = 'simple_view' )
    print 'value:', clt.value    
    clt.configure_traits( view = 'custom_view' )
    print 'value:', clt.value    
