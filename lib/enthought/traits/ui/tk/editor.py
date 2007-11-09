#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: David C. Morrill
# Date: 12/02/2004
# Description: Define the Tkinter Editor base class.
#             
# Symbols defined: Editor
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api    import Int
from enthought.traits.ui.api import Editor as UIEditor

#-------------------------------------------------------------------------------
#  'Editor' class:
#-------------------------------------------------------------------------------

class Editor ( UIEditor ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    layout_style = Int( wx.EXPAND )  # Style for imbedding control in a sizer
    
    #---------------------------------------------------------------------------
    #  Handles the 'control' trait being set:
    #---------------------------------------------------------------------------
    
    def _control_changed ( self, control ):
        """ Handles the 'control' trait being set.
        """
        if control is not None:
            control._editor = self
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        new_value = self.str_value
        var        = self.control.cget( 'textvariable' )
        if var.get() != new_value:
            var.set( new_value )

    #---------------------------------------------------------------------------
    #  Handles an error that occurs while setting the object's trait value:
    #---------------------------------------------------------------------------
 
    def error ( self, excp ):
        """ Handles an error that occurs while setting the object's trait value.
        """
        dlg = wx.MessageDialog( self.control, str( excp ),
                                self.description + ' value error',
                                wx.OK | wx.ICON_INFORMATION )
        dlg.ShowModal()
        dlg.Destroy()
       
    #---------------------------------------------------------------------------
    #  Sets the tooltip for a specified control:
    #---------------------------------------------------------------------------
        
    def set_tooltip ( self, control = None ):
        """ Sets the tooltip for a specified control.
        """
        desc = self.object.base_trait( self.name ).desc
        if desc is not None:
            if control is None:
                control = self.control
            control.SetToolTipString( 'Specifies ' + desc )
            
    #---------------------------------------------------------------------------
    #  Handles the 'enabled' state of the editor being changed:
    #---------------------------------------------------------------------------
            
    def _enabled_changed ( self, enabled ):
        """ Handles the 'enabled' state of the editor being changed.
        """
        self.control.Enable( enabled )
             
