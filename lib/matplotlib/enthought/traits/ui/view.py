#-------------------------------------------------------------------------------
#
#  Define the View class used to represent the structural content of a
#  traits-based user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: View
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits    import Trait, TraitPrefixList, TraitError, Str, Float,\
                                Bool, Instance, Any, Callable
from view_element        import ViewElement, ViewSubElement
from ui                  import UI
from ui_traits           import SequenceTypes, object_trait, style_trait
from handler             import Handler, default_handler
from group               import Group
from item                import Item
from include             import Include

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Name of the view trait:
id_trait = Str( desc = 'the name of the view' )

# Contents of the view trait (i.e. a single Group object):
content_trait = Trait( Group,
                       desc = 'the content of the view' )
                       
# The menu bar for the view:
#menubar_trait = Instance( 'enthought.pyface.action.MenuBarManager',
#                          desc = 'the menu bar for the view' )

# The tool bar for the view:
#toolbar_trait = Instance( 'enthought.pyface.action.ToolBarManager',
#                          desc = 'the tool bar for the view' )
                    
# Reference to a Handler object trait:
handler_trait = Trait( None, Handler,
                       desc = 'the handler for the view' )

# Dialog window title trait:
title_trait = Str( desc = 'the window title for the view' )

# User interface kind trait:
kind_trait = Trait( 'live', 
                    TraitPrefixList( 'panel', 'subpanel', 
                                     'modal', 'nonmodal',
                                     'livemodal', 'live', 'wizard' ), 
                    desc = 'the kind of view window to create',
                    cols = 4 )
           
# Optional window button traits:                    
apply_trait  = Bool( True,
                     desc = "whether to add an 'Apply' button to the view" )
                    
revert_trait = Bool( True,
                     desc = "whether to add a 'Revert' button to the view" )
                    
undo_trait   = Bool( True,
                 desc = "whether to add 'Undo' and 'Redo' buttons to the view" )
          
ok_trait     = Bool( True,
                 desc = "whether to add 'OK' and 'Cancel' buttons to the view" )
          
help_trait   = Bool( True,
                     desc = "whether to add a 'Help' button to the view" )
                     
help_id_trait = Str( desc = "the external help context identifier" )                     
                     
on_apply_trait = Callable( desc = 'the routine to call when modal changes are '
                                  'applied or reverted' )
                     
# Is dialog window resizable trait:
resizable_trait = Bool( False,
                        desc = 'whether dialog can be resized or not' )

# The view position and size traits:                    
width_trait  = Float( -1E6,
                      desc = 'the width of the view window' )
height_trait = Float( -1E6,
                      desc = 'the height of the view window' )
x_trait      = Float( -1E6,
                      desc = 'the x coordinate of the view window' )
y_trait      = Float( -1E6,
                      desc = 'the y coordinate of the view window' )
                    
#-------------------------------------------------------------------------------
#  'View' class:
#-------------------------------------------------------------------------------

class View ( ViewElement ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    id        = id_trait        # The name of the view
    content   = content_trait   # The top-level Group object for the view
    menubar   = Any             # The menu bar for the view
    toolbar   = Any             # The menu bar for the view
#   menubar   = menubar_trait   # The menu bar for the view
#   toolbar   = toolbar_trait   # The tool bar for the view
    handler   = handler_trait   # The Handler object for handling events
    title     = title_trait     # The modal/wizard dialog window title
    kind      = kind_trait      # The kind of user interface to create
    object    = object_trait    # The default object being edited
    style     = style_trait     # The style of user interface to create
    on_apply  = on_apply_trait  # Called when modal changes are applied/reverted
    apply     = apply_trait     # Should an Apply button be added?
    revert    = revert_trait    # Should a Revert button be added?
    undo      = undo_trait      # Should Undo/Redo buttons be added?
    ok        = ok_trait        # Should OK/Cancel buttons be added?
    resizable = resizable_trait # Should dialog be resizable?
    help      = help_trait      # Should a Help button be added?
    help_id   = help_id_trait   # External help context identifier
    x         = x_trait         # Requested view window x coordinate
    y         = y_trait         # Requested view window y coordinate
    width     = width_trait     # Requested view window width
    height    = height_trait    # Requested view window height
    
    # Note: Group objects delegate their 'object' and 'style' traits to the View
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        """ Initializes the object.
        """
        ViewElement.__init__( self, **traits )
        content = []
        accum   = []
        for value in values:
            if isinstance( value, Group ):
                self._flush( content, accum )
                content.append( value )
            elif type( value ) in SequenceTypes:
                self._flush( content, accum )
                content.append( Group( *value ) )
            else:
                accum.append( value )
        self._flush( content, accum )
        
        # If 'content' trait was specified, add it to the end of the content:
        if self.content is not None:
            content.append( self.content )
        
        # Make sure this View is the container for all its children:
        for item in content:
            item.container = self
            
        # Wrap all of the content up into a Group and save it as our content:
        self.content = Group( container = self, *content )

    #---------------------------------------------------------------------------
    #  Creates a UI user interface object:
    #---------------------------------------------------------------------------
    
    def ui ( self, context, parent        = None, 
                            kind          = None, 
                            view_elements = None, 
                            handler       = None ):
        """ Creates a UI user interface object.
        """
        if type( context ) is not dict:
            context = { 'object': context }
        ui = UI( view          = self,
                 context       = context,
                 handler       = handler or self.handler or default_handler(),
                 view_elements = view_elements )
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
        """ Replaces any items which have an 'id' with an Include object with 
            the same 'id', and puts the object with the 'id' into the specified 
            ViewElements object.
        """
        if self.content is not None:
            self.content.replace_include( view_elements )

    #---------------------------------------------------------------------------
    #  Flushes the accumulated Item objects to the contents list as a new Group:
    #---------------------------------------------------------------------------
        
    def _flush ( self, content, accum ):
        """ Flushes the accumulated Item objects to the contents list as a new 
            Group.
        """
        if len( accum ) > 0:
            content.append( Group( *accum ) )
            del accum[:]
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the View:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the View.
        """
        if self.content is None:
            return '[]'
        return "[ %s ]" %  ', '.join( 
               [ item.__repr__() for item in self.content.content ] )
        
