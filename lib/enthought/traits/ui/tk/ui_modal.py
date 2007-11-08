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
# Description: Create a modal Tkinter user interface for a specified UI object.
#             
# Symbols defined: ui_modal
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from ui_panel  import panel, show_help
from constants import screen_dy, scrollbar_dx

#-------------------------------------------------------------------------------
#  Creates a non-modal Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_modal ( ui, parent ):
    ui.control = ModalDialog( ui, parent )
    try:
        ui.prepare_ui()
    except:
        ui.control.Destroy()
        ui.control.ui = None
        ui.control    = None
        ui.result     = False
        raise
    ui.handler.position( ui.info )
    ui.control.ShowModal()
    
#-------------------------------------------------------------------------------
#  'ModalDialog' class:
#-------------------------------------------------------------------------------
    
class ModalDialog ( wx.Dialog ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, ui, parent ):
        wx.Dialog.__init__( self, parent, -1, ui.view.title )
        wx.EVT_CLOSE( self, self._on_cancel )
        wx.EVT_CHAR(  self, self._on_key )
        
        self.ui = ui
        
        # Create the 'context' copies we will need while editing:
        context     = ui.context
        ui._context = context
        ui.context  = self._copy_context( context )
        ui._revert  = self._copy_context( context )
        
        # Create the actual trait sheet panel and imbed it in a scrollable 
        # window:
        sizer       = wx.BoxSizer( wx.VERTICAL )
        sw          = wx.ScrolledWindow( self )
        trait_sheet = panel( ui, sw )
        sizer.Add( trait_sheet, 1, wx.EXPAND | wx.ALL, 4 )
        tsdx, tsdy = trait_sheet.GetSizeTuple()
        tsdx += 8
        tsdy += 8
        
        max_dy = (2 * screen_dy) / 3
        sw.SetAutoLayout( True )
        sw.SetSizer( sizer )
        sw.SetSize( wx.Size( tsdx + ((tsdy > max_dy) * scrollbar_dx), 
                             min( tsdy, max_dy ) ) )
        sw.SetScrollRate( 16, 16 )
        
        sw_sizer = wx.BoxSizer( wx.VERTICAL )
        sw_sizer.Add( sw, 1, wx.EXPAND )
        
        # Create the necessary special function buttons:
        sw_sizer.Add( wx.StaticLine( self, -1 ), 0, wx.EXPAND )
        b_sizer = wx.BoxSizer( wx.HORIZONTAL )
        if ui.view.apply:
            self.apply = self._add_button( 'Apply', self._on_apply, b_sizer, 
                                           False )
            ui.on_trait_change( self._on_applyable, 'modified', 
                                dispatch = 'ui' )
            if ui.view.revert:
                self.revert = self._add_button( 'Revert', self._on_revert, 
                                                b_sizer, False )
        self._add_button( 'OK', self._on_ok, b_sizer )
        self._add_button( 'Cancel', self._on_cancel, b_sizer )
        if ui.view.help:
            self._add_button( 'Help', self._on_help, b_sizer )
        sw_sizer.Add( b_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5 )
         
        # Lay all of the dialog contents out:            
        sw_sizer.Fit( self )
        self.SetSizer( sw_sizer )
        self.SetAutoLayout( True )
        
    #---------------------------------------------------------------------------
    #  Creates a copy of a 'context' dictionary:
    #---------------------------------------------------------------------------
        
    def _copy_context ( self, context ):
        """ Creates a copy of a 'context' dictionary.
        """
        result = {}
        for name, value in context.items():
            result[ name ] = value.clone_traits()
        return result
        
    #---------------------------------------------------------------------------
    #  Applies the traits in the 'from' context to the 'to' context:
    #---------------------------------------------------------------------------
        
    def _apply_context ( self, from_context, to_context ):
        """ Applies the traits in the 'from' context to the 'to' context.
        """
        for name, value in from_context.items():
            to_context[ name ].copy_traits( value )

    #---------------------------------------------------------------------------
    #  Closes the window and saves changes (if allowed by the handler):
    #---------------------------------------------------------------------------
 
    def _on_ok ( self, event = None ):
        """ Closes the window and saves changes (if allowed by the handler).
        """
        if self.ui.handler.close( self.ui.info, True ):
            self._apply_context( self.ui.context, self.ui._context )
            self._close_page( wx.ID_OK )

    #---------------------------------------------------------------------------
    #  Closes the window and discards changes (if allowed by the handler):
    #---------------------------------------------------------------------------
 
    def _on_cancel ( self, event = None ):
        """ Closes the window and discards changes (if allowed by the handler).
        """
        if self.ui.handler.close( self.ui.info, False ):
            self._apply_context( self.ui._revert, self.ui._context )
            self._close_page( wx.ID_CANCEL )
    
    #---------------------------------------------------------------------------
    #  Handles the 'Help' button being clicked:
    #---------------------------------------------------------------------------
           
    def _on_help ( self, event ):
        """ Handles the 'Help' button being clicked.
        """
        show_help( self.ui, event.GetEventObject() )

    #---------------------------------------------------------------------------
    #  Closes the dialog window:
    #---------------------------------------------------------------------------
            
    def _close_page ( self, rc ):            
        """ Closes the dialog window.
        """
        self.EndModal( rc )
        self.ui.control = None
        self.ui.result  = (rc == wx.ID_OK)
        self.ui         = None
        self.Destroy()
 
    #---------------------------------------------------------------------------
    #  Handles the user hitting the 'Esc'ape key:
    #---------------------------------------------------------------------------
 
    def _on_key ( self, event ):
        """ Handles the user hitting the 'Esc'ape key.
        """
        if event.GetKeyCode() == 0x1B:
           self._on_cancel( event )
   
    #---------------------------------------------------------------------------
    #  Handles an 'Apply' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_apply ( self, event ):
        """ Handles an 'Apply' changes request.
        """
        self._apply_context( self.ui.context, self.ui._context )
        self.revert.Enable( True )
        self.ui.modified = False
   
    #---------------------------------------------------------------------------
    #  Handles a 'Revert' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_revert ( self, event ):
        """ Handles a 'Revert' changes request.
        """
        self._apply_context( self.ui._revert, self.ui._context )
        self._apply_context( self.ui._revert, self.ui.context )
        self.revert.Enable( False )
        self.ui.modified = False
            
    #---------------------------------------------------------------------------
    #  Handles the user interface 'modified' state changing:
    #---------------------------------------------------------------------------
            
    def _on_applyable ( self, state ):
        """ Handles the user interface 'modified' state changing.
        """
        self.apply.Enable( state )

    #---------------------------------------------------------------------------
    #  Creates a new dialog button:
    #---------------------------------------------------------------------------

    def _add_button ( self, label, action, sizer, enabled = True ):
        """ Creates a new dialog button.
        """
        button = wx.Button( self, -1, label )
        wx.EVT_BUTTON( self, button.GetId(), action )
        sizer.Add( button, 0, wx.LEFT, 5 )
        button.Enable( enabled )
        return button
               
