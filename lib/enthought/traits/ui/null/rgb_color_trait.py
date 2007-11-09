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
# Data: 02/14/2005
#------------------------------------------------------------------------------
""" Trait definitions for an RGB-based color, which is a tuple of the form 
(*red*, *green*, *blue*), where *red*, *green* and *blue* are floats in the
range from 0.0 to 1.0.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api            import Trait, TraitError
from enthought.traits.trait_base import SequenceTypes
from enthought.traits.ui.api         import RGBColorEditor

#-------------------------------------------------------------------------------
#  Convert a number into an RGB tuple:
#-------------------------------------------------------------------------------

def range_check ( value ):
    """ Checks that *value* can be converted to a value in the range 0.0 to 1.0.
    
    If so, it returns the floating point value; otherwise, it raises a TraitError.
    """
    value = float( value )
    if 0.0 <= value <= 1.0:
        return value
    raise TraitError
    
def convert_to_color ( object, name, value ):
    """ Converts a tuple or an integer to an RGB color value, or raises a 
    TraitError if that is not possible.
    """
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

# RGB versions of standard colors
rgb_standard_colors = {
    'aquamarine':           ( 0.439216, 0.858824, 0.576471 ),
    'black':                ( 0.0, 0.0, 0.0 ),
    'blue':                 ( 0.0, 0.0, 1.0 ),
    'blue violet':          ( 0.623529, 0.372549, 0.623529 ),
    'brown':                ( 0.647059, 0.164706, 0.164706 ),
    'cadet blue':           ( 0.372549, 0.623529, 0.623529 ),
    'coral':                ( 1.0, 0.498039, 0.0 ),
    'cornflower blue':      ( 0.258824, 0.258824, 0.435294 ),
    'cyan':                 ( 0.0, 1.0, 1.0 ),
    'dark grey':            ( 0.184314, 0.184314, 0.184314 ),
    'dark green':           ( 0.184314, 0.309804, 0.184314 ),
    'dark olive green':     ( 0.309804, 0.309804, 0.184314 ),
    'dark orchid':          ( 0.6, 0.196078, 0.8 ),
    'dark slate blue':      ( 0.419608, 0.137255, 0.556863 ),
    'dark slate grey':      ( 0.184314, 0.309804, 0.309804 ),
    'dark turquoise':       ( 0.439216, 0.576471, 0.858824 ),
    'dim grey':             ( 0.329412, 0.329412, 0.329412 ),
    'firebrick':            ( 0.556863, 0.137255, 0.137255 ),
    'forest green':         ( 0.137255, 0.556863, 0.137255 ),
    'gold':                 ( 0.8, 0.498039, 0.196078 ),
    'goldenrod':            ( 0.858824, 0.858824, 0.439216 ),
    'grey':                 ( 0.501961, 0.501961, 0.501961 ),
    'green':                ( 0.0, 1.0, 0.0 ),
    'green yellow':         ( 0.576471, 0.858824, 0.439216 ),
    'indian red':           ( 0.309804, 0.184314, 0.184314 ),
    'khaki':                ( 0.623529, 0.623529, 0.372549 ),
    'light blue':           ( 0.74902, 0.847059, 0.847059 ),
    'light grey':           ( 0.752941, 0.752941, 0.752941 ),
    'light steel':          ( 0.0, 0.0, 0.0 ),
    'blue':                 ( 0.0, 0.0, 1.0 ),
    'lime green':           ( 0.196078, 0.8, 0.196078 ),
    'magenta':              ( 1.0, 0.0, 1.0 ),
    'maroon':               ( 0.556863, 0.137255, 0.419608 ),
    'medium aquamarine':    ( 0.196078, 0.8, 0.6 ),
    'medium blue':          ( 0.196078, 0.196078, 0.8 ),
    'medium forest green':  ( 0.419608, 0.556863, 0.137255 ),
    'medium goldenrod':     ( 0.917647, 0.917647, 0.678431 ),
    'medium orchid':        ( 0.576471, 0.439216, 0.858824 ),
    'medium sea green':     ( 0.258824, 0.435294, 0.258824 ),
    'medium slate blue':    ( 0.498039, 0.0, 1.0 ),
    'medium spring green':  ( 0.498039, 1.0, 0.0 ),
    'medium turquoise':     ( 0.439216, 0.858824, 0.858824 ),
    'medium violet red':    ( 0.858824, 0.439216, 0.576471 ),
    'midnight blue':        ( 0.184314, 0.184314, 0.309804 ),
    'navy':                 ( 0.137255, 0.137255, 0.556863 ),
    'orange':               ( 0.8, 0.196078, 0.196078 ),
    'orange red':           ( 1.0, 0.0, 0.498039 ),
    'orchid':               ( 0.858824, 0.439216, 0.858824 ),
    'pale green':           ( 0.560784, 0.737255, 0.560784 ),
    'pink':                 ( 0.737255, 0.560784, 0.917647 ),
    'plum':                 ( 0.917647, 0.678431, 0.917647 ),
    'purple':               ( 0.690196, 0.0, 1.0 ),
    'red':                  ( 1.0, 0.0, 0.0 ),
    'salmon':               ( 0.435294, 0.258824, 0.258824 ),
    'sea green':            ( 0.137255, 0.556863, 0.419608 ),
    'sienna':               ( 0.556863, 0.419608, 0.137255 ),
    'sky blue':             ( 0.196078, 0.6, 0.8 ),
    'slate blue':           ( 0.0, 0.498039, 1.0 ),
    'spring green':         ( 0.0, 1.0, 0.498039 ),
    'steel blue':           ( 0.137255, 0.419608, 0.556863 ),
    'tan':                  ( 0.858824, 0.576471, 0.439216 ),
    'thistle':              ( 0.847059, 0.74902, 0.847059 ),
    'turquoise':            ( 0.678431, 0.917647, 0.917647 ),
    'violet':               ( 0.309804, 0.184314, 0.309804 ),
    'violet red':           ( 0.8, 0.196078, 0.6 ),
    'wheat':                ( 0.847059, 0.847059, 0.74902 ),
    'white':                ( 1.0, 1.0, 1.0 ),
    'yellow':               ( 1.0, 1.0, 0.0 ),
    'yellow green':         ( 0.6, 0.8, 0.196078 )
}

#-------------------------------------------------------------------------------
#  Define 'null' specific color trait:
#-------------------------------------------------------------------------------
    
# Trait whose value must be an RGB color
RGBColor = Trait( 'white', convert_to_color, rgb_standard_colors, 
                  editor = RGBColorEditor )
       
