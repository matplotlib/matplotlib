#-------------------------------------------------------------------------------
#
#  Define the Include class used to represent a substitutable element within a
#  user interface View.
#
#  Written by: David C. Morrill
#
#  Date: 10/18/2004
#
#  Symbols defined: Include
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits import Str
from view_element     import ViewSubElement
                    
#-------------------------------------------------------------------------------
#  'Include' class:
#-------------------------------------------------------------------------------

class Include ( ViewSubElement ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    id = Str # The name of the substitutable content
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, id, **traits ):
        """ Initializes the object.
        """
        super( ViewSubElement, self ).__init__( **traits )
        self.id = id
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Include:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the Group.
        """
        return "<%s>" % self.id 
    
