#-------------------------------------------------------------------------------
#
#  Define 'factory functions' for all of the standard EditorFactory subclasses.
#
#  Written by: David C. Morrill
#
#  Date: 10/21/2004
#
#  Symbols defined: InstanceEditor
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from toolkit import toolkit

#-------------------------------------------------------------------------------
#  Define 'factory' functions for all of the GUI toolkit dependent traits:
#-------------------------------------------------------------------------------

def ColorTrait ( *args, **traits ):
    return toolkit().color_trait( *args, **traits )

def RGBColorTrait ( *args, **traits ):
    return toolkit().rgb_color_trait( *args, **traits )

def RGBAColorTrait ( *args, **traits ):
    return toolkit().rgba_color_trait( *args, **traits )
    
def FontTrait ( *args, **traits ):
    return toolkit().font_trait( *args, **traits )
    
def KivaFontTrait ( *args, **traits ):
    return toolkit().kiva_font_trait( *args, **traits )
    
#-------------------------------------------------------------------------------
#  Define 'factory' functions for all of the standard EditorFactory subclasses:
#-------------------------------------------------------------------------------
    
def BooleanEditor ( *args, **traits ):
    return toolkit().boolean_editor( *args, **traits )
    
def ButtonEditor ( *args, **traits ):
    return toolkit().button_editor( *args, **traits )
    
def CheckListEditor ( *args, **traits ):
    return toolkit().check_list_editor( *args, **traits )
    
def CodeEditor ( *args, **traits ):
    return toolkit().code_editor( *args, **traits )
    
def ColorEditor ( *args, **traits ):
    return toolkit().color_editor( *args, **traits )
    
def CompoundEditor ( *args, **traits ):
    return toolkit().compound_editor( *args, **traits )
    
def DirectoryEditor ( *args, **traits ):
    return toolkit().directory_editor( *args, **traits )
    
def EnableRGBAColorEditor ( *args, **traits ):
    return toolkit().enable_rgba_color_editor( *args, **traits )
    
def EnumEditor ( *args, **traits ):
    return toolkit().enum_editor( *args, **traits )
    
def FileEditor ( *args, **traits ):
    return toolkit().file_editor( *args, **traits )
    
def FontEditor ( *args, **traits ):
    return toolkit().font_editor( *args, **traits )
    
def KivaFontEditor ( *args, **traits ):
    return toolkit().kiva_font_editor( *args, **traits )
    
def ImageEnumEditor ( *args, **traits ):
    return toolkit().image_enum_editor( *args, **traits )
    
def InstanceEditor ( *args, **traits ):
    return toolkit().instance_editor( *args, **traits )
    
def ListEditor ( *args, **traits ):
    return toolkit().list_editor( *args, **traits )
    
def PlotEditor ( *args, **traits ):
    return toolkit().plot_editor( *args, **traits )
    
def RangeEditor ( *args, **traits ):
    return toolkit().range_editor( *args, **traits )
    
def RGBColorEditor ( *args, **traits ):
    return toolkit().rgb_color_editor( *args, **traits )
    
def RGBAColorEditor ( *args, **traits ):
    return toolkit().rgba_color_editor( *args, **traits )
    
def TextEditor ( *args, **traits ):
    return toolkit().text_editor( *args, **traits )
    
def TreeEditor ( *args, **traits ):
    return toolkit().tree_editor( *args, **traits )
    
def TupleEditor ( *args, **traits ):
    return toolkit().tuple_editor( *args, **traits )
    
