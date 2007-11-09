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
# Date: 10/21/2004
#  Symbols defined: InstanceEditor
#------------------------------------------------------------------------------
""" Defines "factory functions" for all of the standard EditorFactorys
subclasses.

"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from toolkit \
    import toolkit

#-------------------------------------------------------------------------------
#  Define 'factory' functions for all of the GUI toolkit dependent traits:
#-------------------------------------------------------------------------------

def ColorTrait ( *args, **traits ):
    return toolkit().color_trait( *args, **traits )

def RGBColorTrait ( *args, **traits ):
    return toolkit().rgb_color_trait( *args, **traits )

def FontTrait ( *args, **traits ):
    return toolkit().font_trait( *args, **traits )


#-------------------------------------------------------------------------------
#  Define 'factory' functions for all of the standard EditorFactory subclasses:
#-------------------------------------------------------------------------------

def ArrayEditor ( *args, **traits ):
    """ Allows the user to edit 2-D Numeric arrays.
    """
    return toolkit().array_editor( *args, **traits )

def BooleanEditor ( *args, **traits ):
    """ Allows the user to select a true or false condition.
    """
    return toolkit().boolean_editor( *args, **traits )

def ButtonEditor ( *args, **traits ):
    """ Allows the user to click a button; this editor is typically used with
        an event trait to fire the event.
    """
    return toolkit().button_editor( *args, **traits )

def CheckListEditor ( *args, **traits ):
    """ Allows the user to select zero, one, or more values from a finite set of
        possibilities.

        Note that the "simple" style is limited to selecting a single value.
    """
    return toolkit().check_list_editor( *args, **traits )

def CodeEditor ( *args, **traits ):
    """ Allows the user to edit a multi-line string.

        The "simple" and "custom" styles of this editor display multiple lines
        of the string, with line numbers.
    """
    return toolkit().code_editor( *args, **traits )

def ColorEditor ( *args, **traits ):
    """ Allows the user to select a color.
    """
    return toolkit().color_editor( *args, **traits )

def CompoundEditor ( *args, **traits ):
    """ Allows the user to select a value based on a compound trait.

    Because a compound trait is composed of multiple trait definitions, this
    editor factory displays trait editors for each of the constituent traits.
    For example, consider the following trait attribute, defined as a compound
    that accepts integer values in the range of 1 to 6, or text strings
    corresponding to those integers::

        compound = Trait(1, Range(1, 6), 'one', 'two', 'three', 'four',
                            'five', 'six')

    The editor displayed for this trait attribute combines editors for integer
    ranges and for enumerations.
    """
    return toolkit().compound_editor( *args, **traits )

def CustomEditor ( *args, **traits ):
    """ Creates a developer-specified custom editor.
    """
    return toolkit().custom_editor( *args, **traits )

def DirectoryEditor ( *args, **traits ):
    """ Allows the user to specify a directory.
    """
    return toolkit().directory_editor( *args, **traits )

def DropEditor ( *args, **traits ):
    """ Allows dropping an object to set a value.
    """
    return toolkit().drop_editor( *args, **traits )

def DNDEditor ( *args, **traits ):
    """ Allows dragging and dropping an object.
    """
    return toolkit().dnd_editor( *args, **traits )

def EnumEditor ( *args, **traits ):
    """ Allows the user to select a single value from an enumerated list of
    values.
    """
    return toolkit().enum_editor( *args, **traits )

def FileEditor ( *args, **traits ):
    """ Allows the user to select a file.
    """
    return toolkit().file_editor( *args, **traits )

def FontEditor ( *args, **traits ):
    """ Allows the user to select a typeface and type size.
    """
    return toolkit().font_editor( *args, **traits )

def KeyBindingEditor ( *args, **traits ):
    return toolkit().key_binding_editor( *args, **traits )

def HTMLEditor ( *args, **traits ):
    """ Displays formatted HTML text.
    """
    return toolkit().html_editor( *args, **traits )

def ImageEnumEditor ( *args, **traits ):
    """ Allows the user to select an image that represents a value in an
        enumerated list of values.
    """
    return toolkit().image_enum_editor( *args, **traits )

def InstanceEditor ( *args, **traits ):
    """ Allows the user to modify a trait attribute whose value is an instance,
        by modifying the trait attribute values on the instance.
    """
    return toolkit().instance_editor( *args, **traits )

def ListEditor ( *args, **traits ):
    """ Allows the user to modify a list of values.

        The user can add, delete, or reorder items, or change the content of
        items.
    """
    return toolkit().list_editor( *args, **traits )

def NullEditor ( *args, **traits ):
    """ Defines an empty (placeholder) editor.
    """
    return toolkit().null_editor( *args, **traits )

def RangeEditor ( *args, **traits ):
    """ Allows the user to specify a value within a range.
    """
    return toolkit().range_editor( *args, **traits )

def RGBColorEditor ( *args, **traits ):
    return toolkit().rgb_color_editor( *args, **traits )

def SetEditor ( *args, **traits ):
    return toolkit().set_editor( *args, **traits )

def ShellEditor ( *args, **traits ):
    return toolkit().shell_editor( *args, **traits )

def TableEditor ( *args, **traits ):
    """ Allows the user to modify a list of objects using a table editor.
    """
    return toolkit().table_editor( *args, **traits )

def TextEditor ( *args, **traits ):
    """ Allows the user to modify a text string.

        The string value entered by the user is coerced to the appropriate type
        for the trait attribute being modified.
    """
    return toolkit().text_editor( *args, **traits )

def TitleEditor ( *args, **traits ):
    """ Displays a dynamic value using a title control.
    """
    return toolkit().title_editor( *args, **traits )

def TreeEditor ( *args, **traits ):
    """ Allows the user to modify a tree data structure.
    """
    return toolkit().tree_editor( *args, **traits )

def TupleEditor ( *args, **traits ):
    return toolkit().tuple_editor( *args, **traits )

def ValueEditor ( *args, **traits ):
    return toolkit().value_editor( *args, **traits )

