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
# Date: 10/07/2004
#
#  Symbols defined: toolkit
#
#------------------------------------------------------------------------------
""" Defines the stub functions used for creating concrete implementations of
the standard EditorFactory subclasses supplied with the Traits package.

"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasTraits, HasPrivateTraits, TraitError

from ui_traits \
    import SequenceTypes

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# List of implemented UI toolkits
TraitUIToolkits = [ 'wx', 'null' ]

#-------------------------------------------------------------------------------
#  Data:
#-------------------------------------------------------------------------------

# The current GUI toolkit object being used
_toolkit = None

#-------------------------------------------------------------------------------
#  Low-level GUI toolkit selection function:
#-------------------------------------------------------------------------------

def toolkit ( *toolkits ):
    """ Selects and returns a low-level GUI toolkit.

    Use this function to get a reference to the current toolkit.
    """
    global _toolkit

    if len( toolkits ) == 0:
        if _toolkit is not None:
            return _toolkit
        toolkits = TraitUIToolkits
    for toolkit_name in toolkits:
        try:
            package  = 'enthought.traits.ui.' + toolkit_name
            module   = __import__( package )
            _toolkit = getattr( module.traits.ui, toolkit_name ).toolkit
            return _toolkit
        except ImportError:
            pass
    else:
        raise TraitError, ("Could not find any UI toolkit called: %s" %
                           ', '.join( toolkits ))

#-------------------------------------------------------------------------------
#  'Toolkit' class (abstract base class):
#-------------------------------------------------------------------------------

class Toolkit ( HasPrivateTraits ):
    """ Abstract base class for GUI toolkits.
    """
    #---------------------------------------------------------------------------
    #  Create GUI toolkit specific user interfaces using information from the
    #  specified UI object:
    #---------------------------------------------------------------------------

    def ui_panel ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific panel-based user interface using
            information from the specified UI object.
        """
        raise NotImplementedError

    def ui_subpanel ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific subpanel-based user interface using
            information from the specified UI object.
        """
        raise NotImplementedError

    def ui_livemodal ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific modal "live update" dialog user
            interface using information from the specified UI object.
        """
        raise NotImplementedError

    def ui_live ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific non-modal "live update" window user
            interface using information from the specified UI object.
        """
        raise NotImplementedError

    def ui_modal ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific modal dialog user interface using
            information from the specified UI object.
        """
        raise NotImplementedError

    def ui_nonmodal ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific non-modal dialog user interface using
            information from the specified UI object.
        """
        raise NotImplementedError

    def ui_wizard ( self, ui, parent ):
        """ Creates a GUI-toolkit-specific wizard dialog user interface using
            information from the specified UI object.
        """
        raise NotImplementedError

    def view_application ( self, context, view, kind = None, handler = None,
                                      id = '', scrollable = None, args = None ):
        """ Creates a GUI-toolkit-specific modal dialog user interface that
            runs as a complete application using information from the
            specified View object.

        Parameters
        ----------
        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object is
            used.
        view : view or string
            A View object that defines a user interface for editing trait
            attribute values.
        kind : string
            The type of user interface window to create. See the
            **enthought.traits.ui.view.kind_trait** trait for values and
            their meanings. If *kind* is unspecified or None, the **kind**
            attribute of the View object is used.
        handler : Handler object
            A handler object used for event handling in the dialog box. If
            None, the default handler for Traits UI is used.
        id : string
            A unique ID for persisting preferences about this user interface,
            such as size and position. If not specified, no user preferences
            are saved.
        scrollable : Boolean
            Indicates whether the dialog box should be scrollable. When set to
            True, scroll bars appear on the dialog box if it is not large enough
            to display all of the items in the view at one time.

        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Positions the associated dialog window on the display:
    #---------------------------------------------------------------------------

    def position ( self, ui ):
        """ Positions the associated dialog window on the display.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Shows a 'Help' window for a specified UI and control:
    #---------------------------------------------------------------------------

    def show_help ( self, ui, control ):
        """ Shows a Help window for a specified UI and control.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Sets the title for the UI window:
    #---------------------------------------------------------------------------

    def set_title ( self, ui ):
        """ Sets the title for the UI window.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Sets the icon for the UI window:
    #---------------------------------------------------------------------------

    def set_icon ( self, ui ):
        """ Sets the icon for the UI window.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Saves user preference information associated with a UI window:
    #---------------------------------------------------------------------------

    def save_window ( self, ui ):
        """ Saves user preference information associated with a UI window.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Rebuilds a UI after a change to the content of the UI:
    #---------------------------------------------------------------------------

    def rebuild_ui ( self, ui ):
        """ Rebuilds a UI after a change to the content of the UI.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Converts a keystroke event into a corresponding key name:
    #---------------------------------------------------------------------------

    def key_event_to_name ( self, event ):
        """ Converts a keystroke event into a corresponding key name.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Hooks all interesting events for all controls in a ui so that they can
    #  be routed to the corrent event handler:
    #---------------------------------------------------------------------------

    def hook_events ( self, ui, control ):
        """ Hooks all interesting events for all controls in a UI so that they
            can be routed to the correct event handler.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Routes a 'hooked' event to the corrent handler method:
    #---------------------------------------------------------------------------

    def route_event ( self, ui, event ):
        """ Routes a "hooked" event to the corrent handler method.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  GUI toolkit dependent trait definitions:
    #---------------------------------------------------------------------------

    def color_trait ( self, *args, **traits ):
        raise NotImplementedError

    def rgb_color_trait ( self, *args, **traits ):
        raise NotImplementedError

    def font_trait ( self, *args, **traits ):
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  'EditorFactory' factory methods:
    #---------------------------------------------------------------------------

    def array_editor ( self, *args, **traits ):
        raise NotImplementedError

    def boolean_editor ( self, *args, **traits ):
        raise NotImplementedError

    def button_editor ( self, *args, **traits ):
        raise NotImplementedError

    def check_list_editor ( self, *args, **traits ):
        raise NotImplementedError

    def code_editor ( self, *args, **traits ):
        raise NotImplementedError

    def color_editor ( self, *args, **traits ):
        raise NotImplementedError

    def compound_editor ( self, *args, **traits ):
        raise NotImplementedError

    def custom_editor ( self, *args, **traits ):
        raise NotImplementedError

    def directory_editor ( self, *args, **traits ):
        raise NotImplementedError

    def drop_editor ( self, *args, **traits ):
        raise NotImplementedError

    def dnd_editor ( self, *args, **traits ):
        raise NotImplementedError

    def enum_editor ( self, *args, **traits ):
        raise NotImplementedError

    def file_editor ( self, *args, **traits ):
        raise NotImplementedError

    def font_editor ( self, *args, **traits ):
        raise NotImplementedError

    def key_binding_editor ( self, *args, **traits ):
        raise NotImplementedError

    def html_editor ( self, *args, **traits ):
        raise NotImplementedError

    def image_enum_editor ( self, *args, **traits ):
        raise NotImplementedError

    def instance_editor ( self, *args, **traits ):
        raise NotImplementedError

    def list_editor ( self, *args, **traits ):
        raise NotImplementedError

    def null_editor ( self, *args, **traits ):
        raise NotImplementedError

    def ordered_set_editor ( self, *args, **traits ):
        raise NotImplementedError

    def plot_editor ( self, *args, **traits ):
        raise NotImplementedError

    def range_editor ( self, *args, **traits ):
        raise NotImplementedError

    def rgb_color_editor ( self, *args, **traits ):
        raise NotImplementedError

    def shell_editor ( self, *args, **traits ):
        raise NotImplementedError

    def table_editor ( self, *args, **traits ):
        raise NotImplementedError

    def text_editor ( self, *args, **traits ):
        raise NotImplementedError

    def title_editor ( self, *args, **traits ):
        raise NotImplementedError

    def tree_editor ( self, *args, **traits ):
        raise NotImplementedError

    def tuple_editor ( self, *args, **traits ):
        raise NotImplementedError

    def value_editor ( self, *args, **traits ):
        raise NotImplementedError

