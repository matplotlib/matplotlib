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
# Description: Define the Tkinter implementation of the various color editors
#              and the color editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#                   color_trait
#                   clear_color_trait
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api import Trait, TraitFactory, false
from editor_factory   import EditorFactory, SimpleEditor, TextEditor, \
                             ReadonlyEditor
from editor           import Editor
from helper           import position_near

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Standard color samples:
color_choices = ( 0, 128, 192, 255 )
color_samples = [ None ] * 48
i             = 0
for r in color_choices:
    for g in color_choices:
        for b in ( 0, 128, 255 ):
            color_samples[i] = wx.Colour( r, g, b )
            i += 1  

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    mapped = false # Is underlying color trait mapped?
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleColorEditor( parent,
                                  factory     = self, 
                                  ui          = ui, 
                                  object      = object, 
                                  name        = name, 
                                  description = description ) 
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return CustomColorEditor( parent,
                                  factory     = self, 
                                  ui          = ui, 
                                  object      = object, 
                                  name        = name, 
                                  description = description ) 
    
    def text_editor ( self, ui, object, name, description, parent ):
        return TextColorEditor( parent,
                                factory     = self, 
                                ui          = ui, 
                                object      = object, 
                                name        = name, 
                                description = description ) 
    
    def readonly_editor ( self, ui, object, name, description, parent ):
        return ReadonlyColorEditor( parent,
                                    factory     = self, 
                                    ui          = ui, 
                                    object      = object, 
                                    name        = name, 
                                    description = description ) 
       
    #---------------------------------------------------------------------------
    #  Gets the Tkinter color equivalent of the object trait:
    #---------------------------------------------------------------------------
    
    def to_wx_color ( self, editor ):
        """ Gets the Tkinter color equivalent of the object trait.
        """
        if self.mapped:
            return getattr( editor.object, editor.name + '_' )
        else:
            return getattr( editor.object, editor.name )
 
    #---------------------------------------------------------------------------
    #  Gets the application equivalent of a Tkinter value:
    #---------------------------------------------------------------------------
    
    def from_wx_color ( self, color ):
        """ Gets the application equivalent of a Tkinter value.
        """
        return color
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    #---------------------------------------------------------------------------
  
    def str_color ( self, color ):
        """ Returns the text representation of a specified color value.
        """
        if isinstance( color, ( wx.Colour, wx.ColourPtr ) ):
            return "(%d,%d,%d)" % ( color.Red(), color.Green(), color.Blue() )
        return color
                                      
#-------------------------------------------------------------------------------
#  'SimpleColorEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleColorEditor ( SimpleEditor ):
       
    #---------------------------------------------------------------------------
    #  Invokes the pop-up editor for an object trait:
    #---------------------------------------------------------------------------
 
    def popup_editor ( self, event ):
        """ Invokes the pop-up editor for an object trait.
        """
        if not hasattr( self.control, 'is_custom' ):
            # Fixes a problem with the edit field having the focus:
            if self.control.HasCapture():
                self.control.ReleaseMouse()
            self._popup_dialog = ColorDialog( self )
        else:    
            update_handler = self.control.update_handler
            if update_handler is not None:
                update_handler( False )
            color_data = wx.ColourData()
            color_data.SetColour( self.factory.to_wx_color( self ) )
            color_data.SetChooseFull( True )
            dialog = wx.ColourDialog( self.control, color_data )
            if dialog.ShowModal() == wx.ID_OK:
                self.value = self.factory.from_wx_color( 
                                  dialog.GetColourData().GetColour() )
            dialog.Destroy()                      
            if update_handler is not None:
                update_handler( True )
        
    #---------------------------------------------------------------------------
    #  Updates the object trait when a color swatch is clicked:
    #---------------------------------------------------------------------------
        
    def update_object_from_swatch ( self, event ):
        """ Updates the object trait when a color swatch is clicked.
        """
        control    = event.GetEventObject()
        self.value = self.factory.from_wx_color( control.GetBackgroundColour() )
        if control.update_handler is not None:
            control.update_handler()
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        super( SimpleColorEditor, self ).update_editor()
        set_color( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, color ):
        """ Returns the text representation of a specified color value.
        """
        return self.factory.str_color( color ) 
                                      
#-------------------------------------------------------------------------------
#  'CustomColorEditor' class:
#-------------------------------------------------------------------------------
                               
class CustomColorEditor ( SimpleColorEditor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = color_editor_for( self, parent )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        pass
        
#-------------------------------------------------------------------------------
#  'TextColorEditor' class:
#-------------------------------------------------------------------------------
                               
class TextColorEditor ( TextEditor ):

    #---------------------------------------------------------------------------
    #  Handles the user changing the contents of the edit control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        """ Handles the user changing the contents of the edit control.
        """
        self.value = self.control.GetValue()
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        super( TextColorEditor, self ).update_editor()
        set_color( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, color ):
        """ Returns the text representation of a specified color value.
        """
        return self.factory.str_color( color ) 
                                      
#-------------------------------------------------------------------------------
#  'ReadonlyColorEditor' class:
#-------------------------------------------------------------------------------
                               
class ReadonlyColorEditor ( ReadonlyEditor ):
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        super( ReadonlyColorEditor, self ).update_editor()
        set_color( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, color ):
        """ Returns the text representation of a specified color value.
        """
        return self.factory.str_color( color ) 
                                   
#-------------------------------------------------------------------------------
#   Sets the color of the specified editor's color control: 
#-------------------------------------------------------------------------------
                               
def set_color ( editor ):
    """  Sets the color of the specified 'color' control.
    """
    color   = editor.factory.to_wx_color( editor )
    control = editor.control
    control.SetBackgroundColour( color )
    if ((color.Red()   > 192) or
        (color.Blue()  > 192) or
        (color.Green() > 192)):
        control.SetForegroundColour( wx.BLACK )
    else:
        control.SetForegroundColour( wx.WHITE )
    control.Refresh()
       
#----------------------------------------------------------------------------
#  Creates a custom color editor panel for a specified editor:
#----------------------------------------------------------------------------

def color_editor_for ( editor, parent, update_handler = None ):
    """ Creates a custom color editor panel for a specified editor.
    """
    # Create a panel to hold all of the buttons:
    panel = wx.Panel( parent, -1 )
    sizer = wx.BoxSizer( wx.HORIZONTAL )
    swatch_editor = editor.factory.simple_editor( editor.ui, editor.object, 
                                        editor.name, editor.description, panel )
    control = swatch_editor.control
    control.is_custom      = True
    control.update_handler = update_handler
    control.SetSize( wx.Size( 72, 72 ) )
    sizer.Add( control, 1, wx.EXPAND | wx.RIGHT, 4 )
    
    # Add all of the color choice buttons:
    sizer2 = wx.GridSizer( 0, 12, 0, 0 )
    
    for i in range( len( color_samples ) ):
        control = wx.Button( panel, -1, '', size = wx.Size( 18, 18 ) )
        control.SetBackgroundColour( color_samples[i] )
        control.update_handler = update_handler
        wx.EVT_BUTTON( panel, control.GetId(), 
                       swatch_editor.update_object_from_swatch )
        sizer2.Add( control )
        editor.set_tooltip( control )
        
    sizer.Add( sizer2 )
    
    # Set-up the layout:
    panel.SetAutoLayout( True )
    panel.SetSizer( sizer )
    sizer.Fit( panel )
        
    # Return the panel as the result:
    return panel
 
#-------------------------------------------------------------------------------
#  'ColorDialog' class:  
#-------------------------------------------------------------------------------

class ColorDialog ( wx.Frame ):

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
 
    def __init__ ( self, editor ):
        """ Initializes the object.
        """
        wx.Frame.__init__( self, editor.control, -1, '',
                           style = wx.SIMPLE_BORDER )
        wx.EVT_ACTIVATE( self, self._on_close_dialog )
        self._closed    = False
        self._closeable = True
        
        panel = color_editor_for( editor, self, self._close_dialog )
        
        sizer = wx.BoxSizer( wx.VERTICAL )
        sizer.Add( panel )
        self.SetAutoLayout( True )
        self.SetSizer( sizer )
        sizer.Fit( self )
        position_near( editor.control, self )
        self.Show()

    #---------------------------------------------------------------------------
    #  Closes the dialog:
    #---------------------------------------------------------------------------
 
    def _on_close_dialog ( self, event, rc = False ):
        """ Closes the dialog.
        """
        if not event.GetActive():
            self._close_dialog()
  
    #---------------------------------------------------------------------------
    #  Closes the dialog:
    #---------------------------------------------------------------------------
  
    def _close_dialog ( self, closeable = None ):
        """ Closes the dialog.
        """
        if closeable is not None:
            self._closeable = closeable
        if self._closeable and (not self._closed):
            self._closed = True
            self.Destroy()

#-------------------------------------------------------------------------------
#  Convert a number into a wxColour object:
#-------------------------------------------------------------------------------

def convert_to_color ( object, name, value ):
    if isinstance( value, wx.ColourPtr ):
        return wx.Colour( value.Red(), value.Green(), value.Blue() )
    if isinstance( value, wx.Colour ):
        return value
    if type( value ) is int:
        num = int( value )
        return wx.Colour( num / 0x10000, (num / 0x100) & 0xFF, num & 0xFF )
    raise TraitError

convert_to_color.info = ('a wx.Colour instance, an integer which in hex is of '
                         'the form 0xRRGGBB, where RR is red, GG is green, '
                         'and BB is blue')
             
#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

standard_colors = {}
for name in [ 'aquamarine', 'black', 'blue', 'blue violet', 'brown',
              'cadet blue', 'coral', 'cornflower blue', 'cyan', 'dark grey',           
              'dark green', 'dark olive green', 'dark orchid',
              'dark slate blue', 'dark slate grey', 'dark turquoise',
              'dim grey', 'firebrick', 'forest green', 'gold', 'goldenrod',           
              'grey', 'green', 'green yellow', 'indian red', 'khaki', 
              'light blue', 'light grey', 'light steel', 'blue', 'lime green',          
              'magenta', 'maroon', 'medium aquamarine', 'medium blue',
              'medium forest green', 'medium goldenrod', 'medium orchid',
              'medium sea green', 'medium slate blue', 'medium spring green', 
              'medium turquoise', 'medium violet red', 'midnight blue', 'navy',                
              'orange', 'orange red', 'orchid', 'pale green', 'pink', 'plum',                
              'purple', 'red', 'salmon', 'sea green', 'sienna', 'sky blue',
              'slate blue', 'spring green', 'steel blue', 'tan', 'thistle',
              'turquoise', 'violet', 'violet red', 'wheat', 'white', 'yellow',              
              'yellow green' ]:
    try:
        standard_colors[ name ] = convert_to_color( None, None, 
                                                    wx.NamedColour( name ) )
    except:
        pass

#-------------------------------------------------------------------------------
#  Define Tkinter specific color traits:
#-------------------------------------------------------------------------------

# Create a singleton color editor:
color_editor = ToolkitEditorFactory( mapped = True )
    
# Color traits:
color_trait       = Trait( 'white', convert_to_color, standard_colors, 
                           editor = color_editor )
                                 
clear_color_trait = Trait( 'clear', None, convert_to_color, standard_colors, 
                           { 'clear': None }, editor = color_editor )
                           
def TkColor ( value = 'white', **metadata ):
    return Trait( value, color_trait, **metadata )
    
TkColor = TraitFactory( TkColor )    
                           
def TkClearColor ( value = 'white', **metadata ):
    return Trait( value, clear_color_trait, **metadata )
    
TkClearColor = TraitFactory( TkClearColor )    
       
