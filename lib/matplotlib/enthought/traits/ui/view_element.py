#-------------------------------------------------------------------------------
#
#  Define the abstract ViewElement class that all trait view template items
#  (i.e. View, Group, Item) derive from.
#
#  Written by: David C. Morrill
#
#  Date: 10/18/2004
#
#  Symbols defined: ViewElement, ViewSubElement
#
#  (c) Copyright 2004 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import re

from string           import rfind
from matplotlib.enthought.traits import HasStrictTraits, Trait
from ui_traits        import object_trait, style_trait

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

label_pat  = re.compile( r"^(.*)\[(.*)\](.*)$", re.MULTILINE | re.DOTALL )
label_pat2 = re.compile( r"^(.*){(.*)}(.*)$",   re.MULTILINE | re.DOTALL )

#-------------------------------------------------------------------------------
#  'ViewElement' class (abstract):
#-------------------------------------------------------------------------------

class ViewElement ( HasStrictTraits ):
    
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
        pass # Normally overridden in a subclass
            
    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------
            
    def is_includable ( self ):
        """ Returns whether or not the object is replacable by an Include 
            object.
        """
        return False # Normally overridden in a subclass

#-------------------------------------------------------------------------------
#  'DefaultViewElement' class:
#-------------------------------------------------------------------------------

class DefaultViewElement ( ViewElement ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    object = object_trait  # The default context object to edit
    style  = style_trait   # The default editor style to use
                     
#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# The 'container' trait used by ViewSubElements:
container_trait = Trait( DefaultViewElement(), ViewElement )
    
#-------------------------------------------------------------------------------
#  'ViewSubElement' class (abstract):
#-------------------------------------------------------------------------------

class ViewSubElement ( ViewElement ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    container = container_trait # The object this ViewSubElement is contained in
            
    #---------------------------------------------------------------------------
    #  Splits a string at a specified character:
    #---------------------------------------------------------------------------
        
    def _split ( self, name, value, char, finder, assign, result ):
        """ Splits a string at a specified character.
        """
        col = finder( value, char )
        if col < 0:
            return value
        items = ( value[:col].strip(), value[col+1:].strip() )
        if items[ assign ] != '':
            setattr( self, name, items[ assign] )
        return items[ result ]

    #---------------------------------------------------------------------------
    #  Sets an object trait if a specified option string is found:
    #---------------------------------------------------------------------------
        
    def _option ( self, string, option, name, value ):
        col = string.find( option )
        if col >= 0:
            string = string[ : col ] + string[ col + len( option ): ]
            setattr( self, name, value )
        return string

    #---------------------------------------------------------------------------
    #  Parses any of the one character forms of the 'style' trait:
    #---------------------------------------------------------------------------
    
    def _parse_style ( self, value ):
        """ Parses any of the one character forms of the 'style' trait.
        """
        value = self._option( value, '$', 'style', 'simple' )
        value = self._option( value, '@', 'style', 'custom' )
        value = self._option( value, '*', 'style', 'text' )
        value = self._option( value, '~', 'style', 'readonly' )
        value = self._split( 'style',  value, ';', rfind, 1, 0 )
        return value
        
    #---------------------------------------------------------------------------
    #  Parses a '[label]' value from the string definition:
    #---------------------------------------------------------------------------
        
    def _parse_label ( self, value ):
        """ Parses a '[label]' value from the string definition.
        """
        match = label_pat.match( value )
        if match is not None:
            self._parsed_label()
        else:
            match = label_pat2.match( value )
        if match is not None:
            self.label = match.group( 2 )
            value      = match.group( 1 ) + match.group( 3 )
        return value
            
    #---------------------------------------------------------------------------
    #  Handles a label being found in the string definition:
    #---------------------------------------------------------------------------
            
    def _parsed_label ( self ):
        """ Handles a label being found in the string definition.
        """
        pass
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of a specified trait value:
    #---------------------------------------------------------------------------
                                  
    def _repr_value ( self, value, prefix = '', suffix = '', ignore = '' ):
        """ Returns a 'pretty print' version of a specified Item trait value.
        """
        if value == ignore:
            return ''
        return '%s%s%s' % ( prefix, value, suffix )

    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of a single trait:
    #---------------------------------------------------------------------------
                     
    def _repr_option ( self, value, match, result ):
        """ Returns a 'pretty print' version of a single trait.
        """
        if value == match:
            return result
        return ''
        
#-------------------------------------------------------------------------------
#  Patch the main traits module with the correct definition for the ViewElement
#  and ViewSubElement class:
#-------------------------------------------------------------------------------
        
import matplotlib.enthought.traits.has_traits as has_traits
has_traits.ViewElement = ViewElement
    
