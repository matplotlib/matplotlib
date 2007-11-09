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
# Description: Define a subclass of the base Tkinter color editor that
#              represents colors as tuples of the form ( red, green, blue ),
#              where 'red', 'green' and 'blue' are floats in the range from 0.0
#              to 1.0.
#
#  Symbols defined: rgb_color_trait
#                   clear_rgb_color_trait
#                   RGBColor
#                   ClearRGBColor
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import tk

from color_editor                import ToolkitEditorFactory as EditorFactory
from color_editor                import standard_colors
from enthought.traits.api            import Trait, TraitFactory
from enthought.traits.trait_base import SequenceTypes

#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
       
    #---------------------------------------------------------------------------
    #  Gets the Tkinter color equivalent of the object trait:
    #---------------------------------------------------------------------------
    
    def to_wx_color ( self, editor ):
        """ Gets the Tkinter color equivalent of the object trait.
        """
        if self.mapped:
            color = getattr( editor.object, editor.name + '_' )
        else:
            color = getattr( editor.object, editor.name )
        return wx.Colour( int( color[0] * 255.0 ), 
                          int( color[1] * 255.0 ), 
                          int( color[2] * 255.0 ) )
 
    #---------------------------------------------------------------------------
    #  Gets the application equivalent of a Tkinter value:
    #---------------------------------------------------------------------------
    
    def from_wx_color ( self, color ):
        """ Gets the application equivalent of a Tkinter value.
        """
        return ( color.Red()   / 255.0, 
                 color.Green() / 255.0,
                 color.Blue()  / 255.0 )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    #---------------------------------------------------------------------------
  
    def str_color ( self, color ):
        """ Returns the text representation of a specified color value.
        """
        if type( color ) in SequenceTypes:
            return "(%d,%d,%d)" % ( int( color[0] * 255.0 ),
                                    int( color[1] * 255.0 ),
                                    int( color[2] * 255.0 ) )
        return color

#-------------------------------------------------------------------------------
#  Convert a number into a wxColour object:
#-------------------------------------------------------------------------------

def range_check ( value ):
    value = float( value )
    if 0.0 <= value <= 1.0:
        return value
    raise TraitError
    
def convert_to_color ( object, name, value ):
    if (type( value ) in SequenceTypes) and (len( value ) == 3):
        return ( range_check( value[0] ), 
                 range_check( value[1] ), 
                 range_check( value[2] ) )
    if type( value ) is int:
        num = int( value )
        return ( (num / 0x10000)        / 255.0
                 ((num / 0x100) & 0xFF) / 255.0, 
                 (num & 0xFF)           / 255.0 )
    raise TraitError

convert_to_color.info = ('a tuple of the form (r,g,b), where r, g, and b '
    'are floats in the range from 0.0 to 1.0, or an integer which in hex is of '
    'the form 0xRRGGBB, where RR is red, GG is green, and BB is blue')
             
#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

rgb_standard_colors = {}
for name, color in standard_colors.items():
    rgb_standard_colors[ name ] = ( color.Red(  ) / 255.0, 
                                    color.Green() / 255.0,
                                    color.Blue()  / 255.0 )

#-------------------------------------------------------------------------------
#  Define Tkinter specific color traits:
#-------------------------------------------------------------------------------

# Create a singleton color editor:
color_editor = ToolkitEditorFactory( mapped = True )
    
# Color traits:
rgb_color_trait = Trait( 'white', convert_to_color, rgb_standard_colors, 
                         editor = color_editor )
                                 
rgb_clear_color_trait = Trait( 'clear', None, convert_to_color, 
                               rgb_standard_colors, { 'clear': None }, 
                               editor = color_editor )
                           
def RGBColor ( value = 'white', **metadata ):
    return Trait( value, rgb_color_trait, **metadata )
    
RGBColor = TraitFactory( RGBColor )    
                           
def RGBClearColor ( value = 'white', **metadata ):
    return Trait( value, rgb_clear_color_trait, **metadata )
    
RGBClearColor = TraitFactory( RGBClearColor )    
       
