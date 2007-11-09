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
#  Symbols defined: View
#
#------------------------------------------------------------------------------
""" Defines the View class used to represent the structural content of a 
Traits-based user interface.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Trait, TraitPrefixList, TraitError, Str, Float, Bool, Instance, \
           List, Any, Callable, Event, Enum
           
from view_element \
    import ViewElement, ViewSubElement
    
from ui \
    import UI
    
from ui_traits \
    import SequenceTypes, object_trait, style_trait, dock_style_trait, \
           image_trait, export_trait, help_id_trait, buttons_trait
    
from handler \
    import Handler, default_handler
    
from group \
    import Group
    
from item \
    import Item
    
from include \
    import Include

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Name of the view trait
id_trait = Str( desc = 'the name of the view' )

# Contents of the view trait (i.e., a single Group object)
content_trait = Instance( Group,
                          desc = 'the content of the view' )

# The menu bar for the view
#menubar_trait = Instance( 'enthought.pyface.action.MenuBarManager',
#                          desc = 'the menu bar for the view' )

# The tool bar for the view
#toolbar_trait = Instance( 'enthought.pyface.action.ToolBarManager',
#                          desc = 'the tool bar for the view' )

# An optional model/view factory for converting the model into a viewable
# 'model_view' object
model_view_trait = Callable( desc = 'the factory function for converting a' 
                                    'model into a model/view object' )
                    
# Reference to a Handler object trait
handler_trait = Any( desc = 'the handler for the view' )

# Dialog window title trait
title_trait = Str( desc = 'the window title for the view' )

# Dialog window icon trait
#icon_trait = Instance( 'enthought.pyface.image_resource.ImageResource',
#                     desc = 'the ImageResource of the icon file for the view' )

# User interface 'kind' trait. The values have the following meanings:
#
# * 'panel': An embeddable panel. This type of window is intended to be used as
#   part of a larger interface.
# * 'subpanel': An embeddable panel that does not display command buttons,
#   even if the View specifies them.
# * 'modal': A modal dialog box that operates on a clone of the object until 
#   the user commits the change.
# * 'nonmodal':  A nonmodal dialog box that operates on a clone of the object
#   until the user commits the change
# * 'live': A nonmodal dialog box that immediately updates the object.
# * 'livemodal': A modal dialog box that immediately updates the object.
# * 'wizard': A wizard modal dialog box. A wizard contains a sequence of 
#   pages, which can be accessed by clicking **Next** and **Back** buttons. 
#   Changes to attribute values are applied only when the user clicks the
#   **Finish** button on the last page.
kind_trait = Trait( 'live', 
                    TraitPrefixList( 'panel', 'subpanel', 
                                     'modal', 'nonmodal',
                                     'livemodal', 'live', 'wizard' ), 
                    desc = 'the kind of view window to create',
                    cols = 4 )
           
# Traits for optional window buttons

apply_trait  = Bool( True,
                     desc = "whether to add an 'Apply' button to the view" )
                    
revert_trait = Bool( True,
                     desc = "whether to add a 'Revert' button to the view" )
                    
undo_trait   = Bool( True,
                 desc = "whether to add 'Undo' and 'Redo' buttons to the view" )
          
ok_trait     = Bool( True,
                     desc = "whether to add an 'OK' button to the view" )
          
cancel_trait = Bool( True,
                     desc = "whether to add a 'Cancel' button to the view" )
          
help_trait   = Bool( True,
                     desc = "whether to add a 'Help' button to the view" )
                     
on_apply_trait = Callable( desc = 'the routine to call when modal changes are '
                                  'applied or reverted' )
                     
# Is the dialog window is resizable?
resizable_trait = Bool( False,
                        desc = 'whether dialog can be resized or not' )
                     
# Is the view scrollable?
scrollable_trait = Bool( False,
                         desc = 'whether view should be scrollable or not' )

# The valid categories of imported elements that can be dragged into the view
imports_trait = List( Str, desc = 'the categories of elements that can be '
                                  'dragged into the view' )

# The view position and size traits:                    

width_trait  = Float( -1E6,
                      desc = 'the width of the view window' )
height_trait = Float( -1E6,
                      desc = 'the height of the view window' )
x_trait      = Float( -1E6,
                      desc = 'the x coordinate of the view window' )
y_trait      = Float( -1E6,
                      desc = 'the y coordinate of the view window' )
                      
# The result that should be returned if the user clicks the window or dialog 
# close button or icon
close_result_trait = Enum( None, True, False,
                         desc = 'the result to return when the user clicks the '
                                'window or dialog close button or icon' )
                    
#-------------------------------------------------------------------------------
#  'View' class:
#-------------------------------------------------------------------------------

class View ( ViewElement ):
    """ A Traits-based user interface for one or more objects.
    
    The attributes of the View object determine the contents and layout of
    an attribute-editing window. A View object contains a set of Group, 
    Item, and Include objects. A View object can be an attribute of an
    object derived from HasTraits, or it can be a standalone object.
    """
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # A unique identifier for the view.
    id = id_trait
    
    # The top-level Group object for the view
    content = content_trait
    
    # The menu bar for the view. Usually requires a custom **handler**.
    menubar = Any
    
    # The toolbar for the view. Usually requires a custom **handler**.
    toolbar = Any
    
    # List of button actions to add to the view. The **enthought.traits.ui.menu**
    # module defines standard buttons, such as **OKButton**, and standard sets
    # of buttons, such as **ModalButtons**, which can be used to define a value
    # for this attribute. This value can also be a list of button name strings,
    # such as ``['OK', 'Cancel', 'Help']``. If set to the empty list, the 
    # view contains a default set of buttons (equivalent to **LiveButtons**:
    # Undo/Redo, Revert, OK, Cancel, Help). To suppress buttons in the view,
    # use the **NoButtons** variable, defined in **enthought.traits.ui.menu**.
    buttons = buttons_trait
    
    # The menu bar for the view
#   menubar = menubar_trait

    # The tool bar for the view
#   toolbar = toolbar_trait

    # The Handler object that provides GUI logic for handling events in the 
    # window. Set this attribute only if you are using a custom Handler. If
    # not set, the default Traits UI Handler is used.
    handler = handler_trait 
    
    # The factory function for converting a model into a model/view object
    model_view = model_view_trait
    
    # Title for the view, displayed in the title bar when the view appears as a
    # secondary window (i.e., dialog or wizard). If not specified, "Edit
    # properties" is used as the title.
    title = title_trait
    
    # The name of the icon to display in the dialog window title bar
    icon = Any
    
    # The kind of user interface to create
    kind = kind_trait
    
    # The default object being edited
    object = object_trait
    
    # The default editor style of elements in the view.
    style = style_trait
    
    # The default docking style to use for sub-groups of the view. The following
    # values are possible:
    #
    # * 'fixed': No rearrangement of sub-groups is allowed.
    # * 'horizontal': Moveable elements have a visual "handle" to the left by
    #   which the element can be dragged.
    # * 'vertical': Moveable elements have a visual "handle" above them by 
    #   which the element can be dragged.
    # * 'tabbed': Moveable elements appear as tabbed pages, which can be 
    #   arranged within the window or "stacked" so that only one appears at
    #   at a time.
    dock = dock_style_trait
    
    # The image to display on notebook tabs
    image = image_trait
    
    # Called when modal changes are applied or reverted
    on_apply = on_apply_trait
    
    # Should an Apply button be added?  (deprecated -- use *buttons*)
    apply = apply_trait
    
    # Should a Revert button be added?  (deprecated -- use *buttons*)
    revert = revert_trait
    
    # Should Undo/Redo buttons be added?  (deprecated -- use *buttons*)
    undo = undo_trait
    
    # Should an OK button be added?  (deprecated -- use *buttons*)
    ok = ok_trait
    
    # Should a Cancel button be added?  (deprecated -- use *buttons*)
    cancel = cancel_trait
    
    # Can the user resize the window?
    resizable = resizable_trait
    
    # Can the user scroll the view? If set to True, window-level scroll bars
    # appear whenever the window is too small to show all of its contents at
    # one time. If set to False, the window does not scroll, but individual
    # widgets might still contain scroll bars.
    scrollable = scrollable_trait
    
    # The category of exported elements
    export = export_trait
    
    # The valid categories of imported elements
    imports = imports_trait
    
    # Should a Help button be added? (deprecated)
    help = help_trait
    
    # External help context identifier, which can be used by a custom help
    # handler. This attribute is ignored by the default help handler.
    help_id = help_id_trait
    
    # Requested x-coordinate (horizontal position) for the view window. This
    # attribute can be specified in the following ways:
    # 
    # * A positive integer: indicates the number of pixels from the left edge
    #   of the screen to the left edge of the window.
    # * A negative integer: indicates the number of pixels from the right edge
    #   of the screen to the right edge of the window.
    # * A floating point value between 0 and 1: indicates the fraction of the
    #   total screen width between the left edge of the screen and the left edge
    #   of the window.
    # * A floating point value between -1 and 0: indicates the fraction of the
    #   total screen width between the right edge of the screen and the right
    #   edge of the window.
    x = x_trait
    
    # Requested y-coordinate (vertical position) for the view window. This
    # attribute behaves exactly like the **x** attribute, except that its value
    # indicates the position of the top or bottom of the view window relative
    # to the top or bottom of the screen.
    y = y_trait
    
    # Requested width for the view window, as an (integer) number of pixels, or
    # as a (floating point) fraction of the screen width.
    width = width_trait
    
    # Requested height for the view window, as an (integer) number of pixels, or
    # as a (floating point) fraction of the screen height.
    height = height_trait
    
    # Class of dropped objects that can be added
    drop_class = Any
    
    # Event when the view has been updated
    updated = Event
    
    # What result should be returned if the user clicks the window or dialog 
    # close button or icon?
    close_result = close_result_trait
    
    # Note: Group objects delegate their 'object' and 'style' traits to the View
        
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        """ Initializes the object.
        """
        ViewElement.__init__( self, **traits )
        self.set_content( *values )
        
    #---------------------------------------------------------------------------
    #  Sets the content of a view:  
    #---------------------------------------------------------------------------

    def set_content ( self, *values ):
        """ Sets the content of a view.
        """
        content = []
        accum   = []
        for value in values:
            if isinstance( value, ViewSubElement ):
                content.append( value )
            elif type( value ) in SequenceTypes:
                content.append( Group( *value ) )
            else:
                content.append( Item( value ) )
            
        # If there are any 'Item' objects in the content, wrap the content in a
        # Group:
        for item in content:
            if isinstance( item, Item ):
                content = [ Group( *content ) ]
                break
                
        # Wrap all of the content up into a Group and save it as our content:
        self.content = Group( container = self, *content )

    #---------------------------------------------------------------------------
    #  Creates a UI user interface object:
    #---------------------------------------------------------------------------
    
    def ui ( self, context, parent        = None, kind       = None, 
                            view_elements = None, handler    = None,
                            id            = '',   scrollable = None,
                            args          = None ):
        """ Creates a **UI** object, which generates the actual GUI window or
        panel from a set of view elements.
        
        Parameters
        ----------
        context : object or dictionary
            A single object or a dictionary of string/object pairs, whose trait
            attributes are to be edited. If not specified, the current object is
            used.
        parent : window component 
            The window parent of the View object's window
        kind : string
            The kind of window to create. See the **kind_trait** trait for 
            details. If *kind* is unspecified or None, the **kind** attribute
            of the View object is used.
        view_elements : ViewElements object 
            The set of Group, Item, and Include objects contained in the view.
            Do not use this parameter when calling this method directly.
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
        handler = handler or self.handler or default_handler()
        if not isinstance( handler, Handler ):
            handler = handler()
        if args is not None:
            handler.set( **args )
        
        if not isinstance( context, dict ):
            context = context.trait_context()
            
        context.setdefault( 'handler', handler )
                        
        if self.model_view is not None:
            context[ 'object' ] = self.model_view( context[ 'object' ] )
            
        self_id = self.id
        if self_id != '':
            if id != '':
                id = '%s:%s' % ( self_id, id )
            else:
                id = self_id
                
        if scrollable is None:
            scrollable = self.scrollable
            
        ui = UI( view          = self,
                 context       = context,
                 handler       = handler,
                 view_elements = view_elements,
                 title         = self.title,
                 id            = id,
                 scrollable    = scrollable )
                 
        if kind is None:
            kind = self.kind
        
        ui.ui( parent, kind )
        
        return ui
    
    #---------------------------------------------------------------------------
    #  Replaces any items which have an 'id' with an Include object with the 
    #  same 'id', and puts the object with the 'id' into the specified 
    #  ViewElements object: 
    #---------------------------------------------------------------------------
    
    def replace_include ( self, view_elements ):
        """ Replaces any items that have an ID with an Include object with 
            the same ID, and puts the object with the ID into the specified 
            ViewElements object.
        """
        if self.content is not None:
            self.content.replace_include( view_elements )
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the View:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a "pretty print" version of the View.
        """
        if self.content is None:
            return '()'
        return "( %s )" %  ', '.join( 
               [ item.__repr__() for item in self.content.content ] )
        
