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
# Description: Define the Tkinter implementation of the various instance editors
#              and the instance editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api         import HasTraits, Str, Undefined
from enthought.traits.ui.view import kind_trait
from editor_factory           import EditorFactory
from editor                   import Editor
from constants                import scrollbar_dx

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    label = Str         # Optional label for button
    view  = Str         # Optional name of the instance view to use
    kind  = kind_trait  # Kind of pop-up editor (live, modal, nonmodal, wizard)
    
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
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return CustomEditor( parent,
                             factory     = self, 
                             ui          = ui, 
                             object      = object, 
                             name        = name, 
                             description = description ) 
                                      
#-------------------------------------------------------------------------------
#  'SimpleEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = wx.Button( parent, -1, '' )
        wx.EVT_BUTTON( parent, self.control.GetId(), self.edit_instance )
        
    #---------------------------------------------------------------------------
    #  Edit the contents of the object trait when the user clicks the button:
    #---------------------------------------------------------------------------
        
    def edit_instance ( self, event ):
        """ Edit the contents of the object trait when the user clicks the button.
        """
        # Create the user interface:
        ui = self.value.edit_traits( self.factory.view, self.control, 
                                     self.factory.kind )
        
        # Chain our undo history to the new user interface if it does not have
        # its own:
        if ui.history is Undefined:
            ui.history = self.ui.history
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        value = self.value
        if self.factory.label == '':
            label = 'None'
            if value is not None:
                label = value.__class__.__name__
            self.control.SetLabel( label )
        self.control.Enable( isinstance( value, HasTraits ) )
                                      
#-------------------------------------------------------------------------------
#  'CustomEditor' class:
#-------------------------------------------------------------------------------
                               
class CustomEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        # Create a panel to hold the object trait's view:
        self.control = wx.ScrolledWindow( parent, -1 )
        self.control.SetAutoLayout( True )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        panel = self.control
        panel.SetSizer( None )
        panel.DestroyChildren()
        sizer = wx.BoxSizer( wx.VERTICAL )
        value = self.value
        if not isinstance( value, HasTraits ):
            control = wx.StaticText( panel, -1, self.str_value )
        else:
            view     = value.trait_view( self.factory.view )
            self._ui = ui = view.ui( value, panel, 'subpanel' )
            control  = ui.control
            # Chain the sub-panel's undo history to ours:
            ui.history = self.ui.history
        sizer.Add( control, 0, wx.EXPAND )
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        panel.SetScrollRate( 16, 16 )
        width, height = control.GetSize()
        panel.SetSize( wx.Size( width + scrollbar_dx, height ) )
        panel.GetParent().Layout()

