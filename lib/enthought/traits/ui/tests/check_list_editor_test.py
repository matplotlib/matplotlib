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
    import Enum, List
    
from enthought.traits.ui.api \
    import Handler, View, Item, CheckListEditor
    
#-------------------------------------------------------------------------------
#  Constants:  
#-------------------------------------------------------------------------------
        
colors  = [ 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet' ]

numbers = [ 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
            'nine', 'ten' ]
            
#-------------------------------------------------------------------------------
#  'CheckListTest' class:  
#-------------------------------------------------------------------------------
                        
class CheckListTest ( Handler ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
    
    case  = Enum( 'Colors', 'Numbers' )
    value = List( editor = CheckListEditor( values = colors, cols = 5 ) )
    
    #---------------------------------------------------------------------------
    #  Event handlers:  
    #---------------------------------------------------------------------------
        
    def object_case_changed ( self, info ):
        if self.case == 'Colors':
            info.value.factory.values = colors
        else:
            info.value.factory.values = numbers
            
#-------------------------------------------------------------------------------
#  Run the tests:  
#-------------------------------------------------------------------------------
                 
if __name__ == '__main__':
    clt = CheckListTest()
    clt.configure_traits( 
        view = View( 'case', '_', Item( 'value',  id = 'value' ) ) )
    print 'value:', clt.value    
    clt.configure_traits( 
        view = View( 'case', '_', Item( 'value@',  id = 'value' ) ) )
    print 'value:', clt.value    
