#-------------------------------------------------------------------------------
#
#  Define the Item class used to represent a single item within a traits-based 
#  user interface.
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2004
#
#  Symbols defined: Item
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import re

from string                      import find, rfind
from matplotlib.enthought.traits            import Instance, Str, Range, false
from matplotlib.enthought.traits.trait_base import user_name_for
from view_element                import ViewSubElement
from ui_traits                   import container_delegate
from editor_factory              import EditorFactory

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Pattern of all digits:    
all_digits = re.compile( r'\d+' )

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# EditorFactory reference trait:
ItemEditor = Instance( EditorFactory )

# Amount of padding to add around item:
Padding = Range( -15, 15, 0, desc = 'amount of padding to add around item' )

#-------------------------------------------------------------------------------
#  'Item' class:
#-------------------------------------------------------------------------------

class Item ( ViewSubElement ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
   
    id           = Str                # Name of the item
    label        = Str                # User interface label for the item
    name         = Str                # Name of the trait the item is editing
    help         = Str                # Help text describing purpose of item
    object       = container_delegate # Object the item is editing
    style        = container_delegate # Presentation style for the item
    editor       = ItemEditor         # Editor to use for the item
    resizable    = false              # Should the item use extra space?
    defined_when = Str                # Pre-condition for defining the item
    enabled_when = Str                # Pre-condition for enabling the item
    padding      = Padding            # Amount of padding to add around item
   
    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------
   
    def __init__ ( self, value = None, **traits ):
        ViewSubElement.__init__( self, **traits )
        if value is None:
            return
        if not type( value ) is str:
            raise TypeError, ("The argument to Item must be a string of the "
                          "form: {id:}{object.}{name}{[label]}{$|@|*|~|;style}")
        value = self._parse_label( value )
        value = self._parse_style( value )
        value = self._option( value, '#', 'resizable', True )
        value = self._split( 'id',     value, ':', find,  0, 1 )
        value = self._split( 'object', value, '.', find,  0, 1 )
        if value != '':
            self.name = value
            
    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------
            
    def is_includable ( self ):
        """ Returns whether or not the object is replacable by an Include 
            object.
        """
        return (self.id != '')
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the Item represents a spacer or separator:
    #---------------------------------------------------------------------------
        
    def is_spacer ( self ):
        name = self.name.strip()
        return ((name == '') or (name == '_') or 
                (all_digits.match( name ) is not None))
        
    #---------------------------------------------------------------------------
    #  Gets the help text associated with the Item in a specified UI:
    #---------------------------------------------------------------------------
        
    def get_help ( self, ui ):
        """ Gets the help text associated with the Item in a specified UI.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None
           
        # Otherwise, it must be a trait Item:
        if self.help != '':
            return self.help
        return ui.context[ self.object ].base_trait( self.name ).get_help()

    #---------------------------------------------------------------------------
    #  Gets the label to use for a specified Item in a specified UI:
    #---------------------------------------------------------------------------
        
    def get_label ( self, ui ):
        """ Gets the label to use for a specified Item.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None
            
        name   = self.name
        object = ui.context[ self.object ]
        trait  = object.base_trait( name )
        label  = self.label
        if label == '':
            label = user_name_for( name )
        tlabel = trait.label
        if tlabel is None:
            return label
        if type( tlabel ) is str:
            if tlabel[0:3] == '...':
                return label + tlabel[3:]
            if tlabel[-3:] == '...':
                return tlabel[:-3] + label
            if self.label != '':
                return self.label
            return tlabel
        return tlabel( object, name, label )
            
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Item:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the Item.
        """
        return '"%s%s%s%s%s"' % ( self._repr_value( self.id, '', ':' ), 
                                  self._repr_value( self.object, '', '.', 
                                                    'object' ), 
                                  self._repr_value( self.name ),
                                  self._repr_value( self.label,'=' ),
                                  self._repr_value( self.style, ';', '', 
                                                    'simple' ) )

