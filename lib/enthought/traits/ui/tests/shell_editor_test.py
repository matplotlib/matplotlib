#-------------------------------------------------------------------------------
#  
#  Traits UI Python ShellEditor test.  
#  
#  Written by: David C. Morrill
#  
#  Date: 10/13/2005
#  
#  (c) Copyright 2005 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------

from enthought.traits.api         import *
from enthought.traits.ui.api      import *
from enthought.traits.ui.menu import *

#-------------------------------------------------------------------------------
#  'ShellTest' class:  
#-------------------------------------------------------------------------------

class ShellTest ( HasPrivateTraits ):

    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------

    name    = Str
    age     = Int
    weight  = Float
    shell_1 = Str
    shell_2 = Dict
    
    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------
        
    view = View( 'name', 'age', 'weight', '_',
                 Item( 'shell_1', editor = ShellEditor() ),
                 Item( 'shell_2', editor = ShellEditor() ),
                 id        = 'enthought.traits.ui.tests.shell_editor_test',
                 resizable = True,
                 width     = 0.3,
                 height    = 0.3,
                 buttons   = NoButtons )

#-------------------------------------------------------------------------------
#  Run the test:  
#-------------------------------------------------------------------------------
                
if __name__ == '__main__':
    ShellTest().configure_traits()
    
