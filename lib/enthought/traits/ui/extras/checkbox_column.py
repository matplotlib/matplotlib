#------------------------------------------------------------------------------
#
#  Copyright (c) 2006, Enthought, Inc.
#  All rights reserved.
# 
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#  
#  Author: Jason Sugg
#
#  Date: 03/28/2006
#
#  Description: Define the table column descriptor used for toggleable
#               columns.
#
#  Symbols defined: CheckboxColumn
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.pyface.grid.checkbox_renderer \
    import CheckboxRenderer

from enthought.traits.ui.table_column \
    import ObjectColumn

#-------------------------------------------------------------------------------
#  'CheckboxColumn' class:
#-------------------------------------------------------------------------------

class CheckboxColumn ( ObjectColumn ):
    

    #---------------------------------------------------------------------------
    #  Initializes the object:  
    #---------------------------------------------------------------------------
        
    def __init__ ( self, **traits ):
        """ Initializes the object.
        """
        super( CheckboxColumn, self ).__init__( **traits )

        # force the renderer to be a checkbox renderer
        self.renderer = CheckboxRenderer()

    #---------------------------------------------------------------------------
    #  Returns the cell background color for the column for a specified object:  
    #---------------------------------------------------------------------------
    
    def get_cell_color ( self, object ):
        """ Returns the cell background color for the column for a specified 
            object.
        """
        # we override this from the parent class to ALWAYS provide the
        # standard color
        return self.cell_color_

    #---------------------------------------------------------------------------
    #  Returns whether the column is editable for a specified object:  
    #---------------------------------------------------------------------------
                
    def is_editable ( self, object ):
        """ Returns whether the column is editable for a specified object.
        """
        # Although a checkbox column is always editable, we return this
        # to keep a standard editor from appearing. The editing is handled
        # in the renderer's handlers.
        return False
