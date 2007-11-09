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
# Description: Create a 'live update' Tkinter user interface for a specified UI
#              object.
#
#  Symbols defined: ui_live
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from ui_panel                 import panel, show_help
from constants                import screen_dy, scrollbar_dx
from enthought.traits.ui.undo import UndoHistory

#-------------------------------------------------------------------------------
#  Creates a 'live update' Tkinter user interface for a specified UI object:
#-------------------------------------------------------------------------------

def ui_live ( ui, parent ):
    ui.control = LiveWindow( ui, parent )
    try:
        ui.prepare_ui()
    except:
        ui.control.Destroy()
        ui.control.ui = None
        ui.control    = None
        ui.result     = False
        raise
    ui.handler.position( ui.info )
    ui.control.Show()
    
#-------------------------------------------------------------------------------
#  'LiveWindow' class:
#-------------------------------------------------------------------------------
    
class LiveWindow ( wx.Dialog ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, ui, parent ):
        wx.Dialog.__init__( self, parent, -1, ui.view.title )
        wx.EVT_CLOSE( self, self._on_close_page )
        wx.EVT_CHAR(  self, self._on_key )
        
        history = None
        self.ui = ui
        view    = ui.view
        if view.undo or view.revert or view.ok:
            ui.history = history = UndoHistory()
        
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
        
        # Check to see if we need to add any of the special function buttons:
        if (history is not None) or view.help:
            sw_sizer.Add( wx.StaticLine( self, -1 ), 0, wx.EXPAND )
            b_sizer = wx.BoxSizer( wx.HORIZONTAL )
            if view.undo:
                self.undo = self._add_button( 'Undo', self._on_undo, b_sizer, 
                                              False )
                self.redo = self._add_button( 'Redo', self._on_redo, b_sizer, 
                                              False )
                history.on_trait_change( self._on_undoable, 'undoable',
                                         dispatch = 'ui' )
                history.on_trait_change( self._on_redoable, 'redoable',
                                         dispatch = 'ui' )
            if view.revert:
                self.revert = self._add_button( 'Revert', self._on_revert, 
                                                b_sizer, False )
                history.on_trait_change( self._on_revertable, 'undoable',
                                         dispatch = 'ui' )
            if view.ok:
                self._add_button( 'OK', self._on_close_page, b_sizer )
                self._add_button( 'Cancel', self._on_cancel, b_sizer )
            if view.help:
                self._add_button( 'Help', self._on_help, b_sizer )
            sw_sizer.Add( b_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5 )
         
        # Lay all of the dialog contents out:            
        sw_sizer.Fit( self )
        self.SetSizer( sw_sizer )
        self.SetAutoLayout( True )

    #---------------------------------------------------------------------------
    #  Closes the window (if allowed by the handler):
    #---------------------------------------------------------------------------
 
    def _on_close_page ( self, event = None ):
        """ Closes the window (if allowed by the handler).
        """
        if self.ui.handler.close( self.ui.info, True ):
            self._close_page()

    #---------------------------------------------------------------------------
    #  Closes the dialog window:
    #---------------------------------------------------------------------------
            
    def _close_page ( self ):            
        """ Closes the dialog window.
        """
        self.ui.control = None
        self.ui.result  = True
        self.ui         = None
        self.Destroy()
 
    #---------------------------------------------------------------------------
    #  Handles the user hitting the 'Esc'ape key:
    #---------------------------------------------------------------------------
 
    def _on_key ( self, event ):
        """ Handles the user hitting the 'Esc'ape key.
        """
        if event.GetKeyCode() == 0x1B:
           self._on_close_page( event )
   
    #---------------------------------------------------------------------------
    #  Handles an 'Undo' change request:
    #---------------------------------------------------------------------------
           
    def _on_undo ( self, event ):
        """ Handles an 'Undo' change request.
        """
        self.ui.history.undo()
   
    #---------------------------------------------------------------------------
    #  Handles a 'Redo' change request:
    #---------------------------------------------------------------------------
           
    def _on_redo ( self, event ):
        """ Handles a 'Redo' change request.
        """
        self.ui.history.redo()
   
    #---------------------------------------------------------------------------
    #  Handles a 'Revert' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_revert ( self, event ):
        """ Handles a 'Revert' all changes request.
        """
        self.ui.history.revert()
   
    #---------------------------------------------------------------------------
    #  Handles a 'Cancel' all changes request:
    #---------------------------------------------------------------------------
           
    def _on_cancel ( self, event ):
        """ Handles a 'Cancel' all changes request.
        """
        if self.ui.handler.close( self.ui.info, True ):
            self._on_revert( event )
            self._close_page()
    
    #---------------------------------------------------------------------------
    #  Handles the 'Help' button being clicked:
    #---------------------------------------------------------------------------
           
    def _on_help ( self, event ):
        """ Handles the 'Help' button being clicked.
        """
        show_help( self.ui, event.GetEventObject() )
            
    #---------------------------------------------------------------------------
    #  Handles the undo history 'undoable' state changing:
    #---------------------------------------------------------------------------
            
    def _on_undoable ( self, state ):
        """ Handles the undo history 'undoable' state changing.
        """
        self.undo.Enable( state )
            
    #---------------------------------------------------------------------------
    #  Handles the undo history 'redoable' state changing:
    #---------------------------------------------------------------------------
            
    def _on_redoable ( self, state ):
        """ Handles the undo history 'redoable' state changing.
        """
        self.redo.Enable( state )
            
    #---------------------------------------------------------------------------
    #  Handles the 'revert' state changing:
    #---------------------------------------------------------------------------
            
    def _on_revertable ( self, state ):
        """ Handles the 'revert' state changing.
        """
        self.revert.Enable( state )

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
           
