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
# Description: Define the Tkinter implementation of the various font editors and
#              the font editor factory.
#
#  Symbols defined: ToolkitEditorFactory
#                   font_trait
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from enthought.traits.api import Trait, TraitHandler, TraitError, TraitFactory
from editor_factory   import EditorFactory, SimpleEditor, TextEditor, \
                             ReadonlyEditor
from editor           import Editor
from helper           import choice_width
     
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Standard font point sizes:
PointSizes = [
   '8',  '9', '10', '11', '12', '14', '16', '18', 
  '20', '22', '24', '26', '28', '36', '48', '72'
]            

# All available font facenames:
facenames = None

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, name, description, parent ):
        return SimpleFontEditor( parent,
                                 factory     = self, 
                                 ui          = ui, 
                                 object      = object, 
                                 name        = name, 
                                 description = description ) 
    
    def custom_editor ( self, ui, object, name, description, parent ):
        return CustomFontEditor( parent,
                                 factory     = self, 
                                 ui          = ui, 
                                 object      = object, 
                                 name        = name, 
                                 description = description ) 
    
    def text_editor ( self, ui, object, name, description, parent ):
        return TextFontEditor( parent,
                               factory     = self, 
                               ui          = ui, 
                               object      = object, 
                               name        = name, 
                               description = description ) 
    
    def readonly_editor ( self, ui, object, name, description, parent ):
        return ReadonlyFontEditor( parent,
                                   factory     = self, 
                                   ui          = ui, 
                                   object      = object, 
                                   name        = name, 
                                   description = description ) 
   
    #---------------------------------------------------------------------------
    #  Returns a wxFont object corresponding to a specified object's font trait:
    #---------------------------------------------------------------------------
    
    def to_wx_font ( self, editor ):
        """ Returns a wxFont object corresponding to a specified object's font 
            trait.
        """
        font = editor.value
        return wx.Font( font.GetPointSize(), font.GetFamily(), font.GetStyle(),
                        font.GetWeight(),    font.GetUnderlined(), 
                        font.GetFaceName() )
 
    #---------------------------------------------------------------------------
    #  Gets the application equivalent of a Tkinter Font value:
    #---------------------------------------------------------------------------
    
    def from_wx_font ( self, font ):
        """ Gets the application equivalent of a Tkinter Font value.
        """
        return font

    #---------------------------------------------------------------------------
    #  Returns the text representation of the specified object trait value:
    #---------------------------------------------------------------------------
    
    def str_font ( self, font ):
        """ Returns the text representation of the specified object trait value.
        """
        weight = { wx.LIGHT: ' Light',
                   wx.BOLD:  ' Bold'   }.get( font.GetWeight(), '' )
        style  = { wx.SLANT: ' Slant',
                   wx.ITALIC:' Italic' }.get( font.GetStyle(), '' )
        return '%s point %s%s%s' % (
               font.GetPointSize(), font.GetFaceName(), style, weight )
                                      
#-------------------------------------------------------------------------------
#  'SimpleFontEditor' class:
#-------------------------------------------------------------------------------
                               
class SimpleFontEditor ( SimpleEditor ):
       
    #---------------------------------------------------------------------------
    #  Invokes the pop-up editor for an object trait:
    #---------------------------------------------------------------------------
 
    def popup_editor ( self, event ):
        """ Invokes the pop-up editor for an object trait.
        """
        font_data = wx.FontData()
        font_data.SetInitialFont( self.factory.to_wx_font( self ) )
        dialog = wx.FontDialog( self.control, font_data )
        if dialog.ShowModal() == wx.ID_OK:
            self.value = self.factory.from_wx_font(
                              dialog.GetFontData().GetChosenFont() )
        dialog.Destroy()
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        super( SimpleFontEditor, self ).update_editor()
        set_font( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified font value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, font ):
        """ Returns the text representation of a specified font value.
        """
        return self.factory.str_font( font ) 
                                      
#-------------------------------------------------------------------------------
#  'CustomFontEditor' class:
#-------------------------------------------------------------------------------
                               
class CustomFontEditor ( Editor ):
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        # Create a panel to hold all of the buttons:
        self.control = panel = wx.Panel( parent, -1 )
        sizer = wx.BoxSizer( wx.VERTICAL )
        
        # Add the standard font control:
        font = self._font = wx.TextCtrl( panel, -1, self.str_value )
        wx.EVT_KILL_FOCUS( font, self.update_object )
        wx.EVT_TEXT_ENTER( panel, font.GetId(), self.update_object )
        sizer.Add( font, 0, wx.EXPAND | wx.BOTTOM, 3 )
        
        # Add all of the font choice controls:
        sizer2    = wx.BoxSizer( wx.HORIZONTAL )
        facenames = all_facenames()
        control   = self._facename = wx.Choice( panel, -1, wx.Point( 0, 0 ), 
                          wx.Size( choice_width( facenames ), 20 ), 
                          facenames )
                        
        sizer2.Add( control, 2, wx.EXPAND )
        wx.EVT_CHOICE( panel, control.GetId(), self.update_object_parts )
        
        control = self._point_size = wx.Choice( panel, -1, 
                                        wx.Point( 0, 0 ), wx.Size( 30, 20 ), 
                                        PointSizes )
        sizer2.Add( control, 1, wx.EXPAND | wx.RIGHT, 3 )
        wx.EVT_CHOICE( panel, control.GetId(), self.update_object_parts ) 
           
        sizer.Add( sizer2, 0, wx.EXPAND )
        
        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

    #---------------------------------------------------------------------------
    #  Handles the user changing the contents of the font text control:
    #---------------------------------------------------------------------------
  
    def update_object ( self, event ):
        """ Handles the user changing the contents of the font text control.
        """
        self.value = self._font.GetValue()
           
    #---------------------------------------------------------------------------
    #  Handles the user modifying one of the font components:
    #---------------------------------------------------------------------------
  
    def update_object_parts ( self, event ):
        """ Handles the user modifying one of the font components.
        """
        point_size = int( self._point_size.GetStringSelection() )
        facename   = self._facename.GetStringSelection()
        font       = wx.Font( point_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL,
                              faceName = facename )
        self.value = self.factory.from_wx_font( font )
         
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        font = self.factory.to_wx_font( self )
        try:
           self._facename.SetStringSelection( font.GetFaceName() )
        except:
           self._facename.SetSelection( 0 )
        try:
           self._point_size.SetStringSelection( str( font.GetPointSize() ) )
        except:
           self._point_size.SetSelection( 0 )
        font.SetPointSize( min( 10, font.GetPointSize() ) )
        self._font.SetValue( self.str_value )
        self._font.SetFont( font )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified font value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, font ):
        """ Returns the text representation of a specified font value.
        """
        return self.factory.str_font( font ) 
        
#-------------------------------------------------------------------------------
#  'TextFontEditor' class:
#-------------------------------------------------------------------------------
                               
class TextFontEditor ( TextEditor ):

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
        super( TextFontEditor, self ).update_editor()
        set_font( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified font value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, font ):
        """ Returns the text representation of a specified font value.
        """
        return self.factory.str_font( font ) 
                                      
#-------------------------------------------------------------------------------
#  'ReadonlyFontEditor' class:
#-------------------------------------------------------------------------------
                               
class ReadonlyFontEditor ( ReadonlyEditor ):
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        super( ReadonlyFontEditor, self ).update_editor()
        set_font( self )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified font value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, font ):
        """ Returns the text representation of a specified font value.
        """
        return self.factory.str_font( font ) 

#-------------------------------------------------------------------------------
#  Set the editor control's font to match a specified font: 
#-------------------------------------------------------------------------------
        
def set_font ( editor ):
    font = editor.factory.to_wx_font( editor )
    font.SetPointSize( min( 10, font.GetPointSize() ) )
    editor.control.SetFont( font )
        
#-------------------------------------------------------------------------------
#  Returns a list of all available font facenames:
#-------------------------------------------------------------------------------

def all_facenames ( ):
    """ Returns a list of all available font facenames.
    """
    global facenames
    
    if facenames is None:
        facenames = FontEnumerator().facenames()
        facenames.sort()
    return facenames

#-------------------------------------------------------------------------------
#  'FontEnumerator' class:
#-------------------------------------------------------------------------------

class FontEnumerator ( wx.FontEnumerator ):
    
    #---------------------------------------------------------------------------
    #  Returns a list of all available font facenames:
    #---------------------------------------------------------------------------
    
    def facenames ( self ):
        """ Returns a list of all available font facenames.
        """
        self._facenames = []
        self.EnumerateFacenames()
        return self._facenames
        
    #---------------------------------------------------------------------------
    #  Adds a facename to the list of facenames:
    #---------------------------------------------------------------------------
    
    def OnFacename ( self, facename ):
        """ Adds a facename to the list of facenames.
        """
        self._facenames.append( facename )
        return True

#-------------------------------------------------------------------------------
#  Convert a string into a valid 'wxFont' object (if possible):
#-------------------------------------------------------------------------------

font_families = {
    'default':    wx.DEFAULT,
    'decorative': wx.DECORATIVE,
    'roman':      wx.ROMAN,
    'script':     wx.SCRIPT,
    'swiss':      wx.SWISS,
    'modern':     wx.MODERN
}

font_styles = {
    'slant':  wx.SLANT,
    'italic': wx.ITALIC
}

font_weights = {
    'light': wx.LIGHT,
    'bold':  wx.BOLD
}

font_noise = [ 'pt', 'point', 'family' ]

#-------------------------------------------------------------------------------
#  'TraitWXFont' class'
#-------------------------------------------------------------------------------

class TraitWXFont ( TraitHandler ):
    
    #---------------------------------------------------------------------------
    #  Validates that the value is a valid font:
    #---------------------------------------------------------------------------
    
    def validate ( self, object, name, value ):
        """ Validates that the value is a valid font.
        """
        if isinstance( value, wx.FontPtr ):
            return wx.Font( value.GetPointSize(),  value.GetFamily(), 
                            value.GetStyle(),      value.GetWeight(), 
                            value.GetUnderlined(), value.GetFaceName() )
        if isinstance( value, wx.Font ):
            return value
        try:
            point_size = 10
            family     = wx.DEFAULT
            style      = wx.NORMAL
            weight     = wx.NORMAL
            underline  = 0
            facename   = []
            for word in value.split():
                lword = word.lower()
                if font_families.has_key( lword ):
                    family = font_families[ lword ]
                elif font_styles.has_key( lword ):
                    style = font_styles[ lword ]
                elif font_weights.has_key( lword ):
                    weight = font_weights[ lword ]
                elif lword == 'underline':
                    underline = 1
                elif lword not in font_noise:
                    try:
                        point_size = int( lword )
                    except:
                        facename.append( word )
            return wx.Font( point_size, family, style, weight, underline,
                            ' '.join( facename ) )
        except:
            pass
        raise TraitError, ( object, name, 'a font descriptor string',
                            repr( value ) )

    def info ( self ):                              
        return ( "a string describing a font (e.g. '12 pt bold italic "
                 "swiss family Arial' or 'default 12')" )

#-------------------------------------------------------------------------------
#  Define a Tkinter specific font trait:
#-------------------------------------------------------------------------------

fh = TraitWXFont()
font_trait = Trait( fh.validate( None, None, 'Arial 10' ), fh, 
                    editor = ToolkitEditorFactory() )
        
def TkFont ( value = 'Arial 10', **metadata ):
    return Trait( value, font_trait, **metadata )
    
TkFont = TraitFactory( TkFont )    
    
