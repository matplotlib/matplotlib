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
# Date: 02/14/2005
#------------------------------------------------------------------------------
""" Trait definition for a null-based (i.e., no UI) color.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api    import Trait, TraitError
from enthought.traits.ui.api import ColorEditor

#-------------------------------------------------------------------------------
#  Convert a number into a wxColour object:
#-------------------------------------------------------------------------------

def convert_to_color ( object, name, value ):
    """ Converts a number into a wxColour object.
    """
    if type( value ) is int:
        return num & 0xFFFFFF
    raise TraitError

convert_to_color.info = ('an integer which in hex is of the form 0xRRGGBB, '
                         'where RR is red, GG is green, and BB is blue')
             
#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

standard_colors = {
    'aquamarine':          0x70DB93,
    'black':               0x000000,
    'blue':                0x0000FF,
    'blue violet':         0x9F5F9F,
    'brown':               0xA52A2A,
    'cadet blue':          0x5F9F9F,
    'coral':               0xFF7F00,
    'cornflower blue':     0x42426F,
    'cyan':                0x00FFFF,
    'dark grey':           0x2F2F2F,
    'dark green':          0x2F4F2F,
    'dark olive green':    0x4F4F2F,
    'dark orchid':         0x9932CC,
    'dark slate blue':     0x6B238E,
    'dark slate grey':     0x2F4F4F,
    'dark turquoise':      0x7093DB,
    'dim grey':            0x545454,
    'firebrick':           0x8E2323,
    'forest green':        0x238E23,
    'gold':                0xCC7F32,
    'goldenrod':           0xDBDB70,
    'grey':                0x808080,
    'green':               0x00FF00,
    'green yellow':        0x93DB70,
    'indian red':          0x4F2F2F,
    'khaki':               0x9F9F5F,
    'light blue':          0xBFD8D8,
    'light grey':          0xC0C0C0,
    'light steel':         0x000000,
    'blue':                0x0000FF,
    'lime green':          0x32CC32,
    'magenta':             0xFF00FF,
    'maroon':              0x8E236B,
    'medium aquamarine':   0x32CC99,
    'medium blue':         0x3232CC,
    'medium forest green': 0x6B8E23,
    'medium goldenrod':    0xEAEAAD,
    'medium orchid':       0x9370DB,
    'medium sea green':    0x426F42,
    'medium slate blue':   0x7F00FF,
    'medium spring green': 0x7FFF00,
    'medium turquoise':    0x70DBDB,
    'medium violet red':   0xDB7093,
    'midnight blue':       0x2F2F4F,
    'navy':                0x23238E,
    'orange':              0xCC3232,
    'orange red':          0xFF007F,
    'orchid':              0xDB70DB,
    'pale green':          0x8FBC8F,
    'pink':                0xBC8FEA,
    'plum':                0xEAADEA,
    'purple':              0xB000FF,
    'red':                 0xFF0000,
    'salmon':              0x6F4242,
    'sea green':           0x238E6B,
    'sienna':              0x8E6B23,
    'sky blue':            0x3299CC,
    'slate blue':          0x007FFF,
    'spring green':        0x00FF7F,
    'steel blue':          0x236B8E,
    'tan':                 0xDB9370,
    'thistle':             0xD8BFD8,
    'turquoise':           0xADEAEA,
    'violet':              0x4F2F4F,
    'violet red':          0xCC3299,
    'wheat':               0xD8D8BF,
    'white':               0xFFFFFF,
    'yellow':              0xFFFF00,
    'yellow green':        0x99CC32
}

#-------------------------------------------------------------------------------
#  Define 'null' specific color traits:
#-------------------------------------------------------------------------------
    
# Color traits
NullColor = Trait( 'white', convert_to_color, standard_colors, 
                   editor = ColorEditor )
       
