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
# Description: Define the Tkinter implementation of the various file editors and
#              the file editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from os.path          import abspath
from enthought.traits.api import List, Str, false
from text_editor      import ToolkitEditorFactory as EditorFactory
from text_editor      import SimpleEditor         as SimpleTextEditor

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Wildcard filter:
filter_trait = List( Str )

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    filter       = filter_trait  # Wildcard filter to apply to the file dialog
    truncate_ext = false      # Should file extension be truncated?
    auto_set     = false      # Is user input set on every keystroke? (override)
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description ) 
                                      
#-------------------------------------------------------------------------------
#  'SimpleEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleEditor ( SimpleTextEditor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = panel = wx.Panel( parent, -1 )
        sizer        = wx.BoxSizer( wx.HORIZONTAL )
        if self.factory.enter_set:
            control = wx.TextCtrl( panel, -1, '', style = wx.TE_PROCESS_ENTER )
            wx.EVT_TEXT_ENTER( panel, control.GetId(), self.update_object )
        else:
            control = wx.TextCtrl( panel, -1, '' )
        self._filename = control
        wx.EVT_KILL_FOCUS( control, self.update_object )
        if self.factory.auto_set:
            wx.EVT_TEXT( panel, control.GetId(), self.update_object )
        sizer.Add( control, 1, wx.EXPAND | wx.ALIGN_CENTER )
        button = wx.Button( panel, -1, 'Browse...' )
        sizer.Add( button, 0, wx.LEFT | wx.ALIGN_CENTER, 8 )
        wx.EVT_BUTTON( panel, button.GetId(), self.show_file_dialog )
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

    #---------------------------------------------------------------------------
    #  Handles the user changing the contents of the edit control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        """ Handles the user changing the contents of the edit control.
        """
        try:
            filename = self._filename.GetValue()
            if self.factory.truncate_ext:
                filename = splitext( filename )[0] 
            self.value = filename
        except TraitError, excp:
            pass
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        self._filename.SetValue( self.str_value )
       
    #---------------------------------------------------------------------------
    #  Displays the pop-up file dialog:
    #---------------------------------------------------------------------------
 
    def show_file_dialog ( self, event ):
        """ Displays the pop-up file dialog.
        """
        dlg      = self.create_file_dialog()
        rc       = (dlg.ShowModal() == wx.ID_OK)
        filename = abspath( dlg.GetPath() )
        dlg.Destroy()
        if rc:
            if self.factory.truncate_ext:
               filename = splitext( filename )[0] 
            self.value = filename
            self.update_editor()

    #---------------------------------------------------------------------------
    #  Creates the correct type of file dialog:
    #---------------------------------------------------------------------------
           
    def create_file_dialog ( self ):
        """ Creates the correct type of file dialog.
        """
        dlg = wx.FileDialog( self.control, message = 'Select a File' )
        dlg.SetFilename( self._filename.GetValue() )
        if len( self.factory.filter ) > 0:
            dlg.SetWildcard( '|'.join( self.factory.filter[:] ) )
        return dlg

