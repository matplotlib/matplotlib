#-------------------------------------------------------------------------------
#  
#  Trait definition for an RGBA-based color.
#  
#  Written by: David C. Morrill
#  
#  Data: 12/22/2004
#  
#  (c) Copyright 2004 by Enthought, Inc.
#  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits            import Trait, TraitError
from matplotlib.enthought.traits.trait_base import SequenceTypes
from matplotlib.enthought.traits.ui         import RGBAColorEditor

from rgb_color_trait             import rgb_standard_colors

#-------------------------------------------------------------------------------
#  Convert a value into an Enable/Kiva color:
#-------------------------------------------------------------------------------

def convert_to_color ( object, name, value ):
    if ((type( value ) in SequenceTypes) and
        (len( value ) == 4) and
        (0.0 <= value[0] <= 1.0) and
        (0.0 <= value[1] <= 1.0) and
        (0.0 <= value[2] <= 1.0) and
        (0.0 <= value[3] <= 1.0)):
        return value
    if type( value ) is int:
        return (  (value / 0x1000000)       / 255.0, 
                 ((value / 0x10000) & 0xFF) / 255.0, 
                 ((value / 0x100)   & 0xFF) / 255.0,
                  (value & 0xFF)            / 255.0 )
    raise TraitError

convert_to_color.info = ('a tuple of the form (red,green,blue,alpha), where '
                         'each component is in the range from 0.0 to 1.0, or '
                         'an integer which in hex is of the form 0xAARRGGBB, '
                         'where AA is alpha, RR is red, GG is green, and BB is '
                         'blue')
             
#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

rgba_standard_colors = {}
for name, color in rgb_standard_colors.items():
    rgba_standard_colors[ name ] = ( color[0], color[1], color[1], 1.0 )
rgba_standard_colors[ 'clear' ] = ( 0.0, 0.0, 0.0, 0.0 )

#-------------------------------------------------------------------------------
#  Define Enable/Kiva specific color trait:
#-------------------------------------------------------------------------------
    
# Color trait:
RGBAColor = Trait( 'white', convert_to_color, rgba_standard_colors, 
                            editor = RGBAColorEditor )

