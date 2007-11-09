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
# Date: 10/18/2004
#
#  Symbols defined: ViewElement, ViewSubElement
#
#------------------------------------------------------------------------------
""" Defines the abstract ViewElement class that all trait view template items
(i.e., View, Group, Item, Include) derive from.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import re

from string \
    import rfind
    
from enthought.traits.api \
    import HasPrivateTraits, Trait, true
    
from ui_traits \
    import object_trait, style_trait, dock_style_trait, image_trait, \
           export_trait, help_id_trait

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

label_pat  = re.compile( r"^(.*)\[(.*)\](.*)$", re.MULTILINE | re.DOTALL )
label_pat2 = re.compile( r"^(.*){(.*)}(.*)$",   re.MULTILINE | re.DOTALL )

#-------------------------------------------------------------------------------
#  'ViewElement' class (abstract):
#-------------------------------------------------------------------------------

class ViewElement ( HasPrivateTraits ):
    """ An element of a view.
    """
    #---------------------------------------------------------------------------
    #  Replaces any items which have an 'id' with an Include object with the 
    #  same 'id', and puts the object with the 'id' into the specified 
    #  ViewElements object: 
    #---------------------------------------------------------------------------
    
    def replace_include ( self, view_elements ):
        """ Searches the current object's **content** attribute for objects that
        have an **id** attribute, and replaces each one with an Include object 
        with the same **id** value, and puts the replaced object into the 
        specified ViewElements object.
        
        Parameters
        ----------
        view_elements : ViewElements object
            Object containing Group, Item, and Include objects
        """
        pass # Normally overridden in a subclass
            
    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------
            
    def is_includable ( self ):
        """ Returns whether the object is replacable by an Include object.
        """
        return False # Normally overridden in a subclass

#-------------------------------------------------------------------------------
#  'DefaultViewElement' class:
#-------------------------------------------------------------------------------

class DefaultViewElement ( ViewElement ):
    """ A view element that can be used as a default value for traits whose
    value is a view element.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # The default context object to edit
    object = object_trait
    
    # The default editor style to use
    style = style_trait   
    
    # The default dock style to use
    dock = dock_style_trait
    
    # The default notebook tab image to use                        
    image = image_trait
    
    # The category of elements dragged out of the view
    export = export_trait
    
    # Should labels be added to items in a group?
    show_labels = true
                     
#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# The container trait used by ViewSubElements
container_trait = Trait( DefaultViewElement(), ViewElement )
    
#-------------------------------------------------------------------------------
#  'ViewSubElement' class (abstract):
#-------------------------------------------------------------------------------

class ViewSubElement ( ViewElement ):
    """ Abstract class representing elements that can be contained in a view.
    """
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # The object this ViewSubElement is contained in; must be a ViewElement.
    container = container_trait 
    # External help context identifier
    help_id   = help_id_trait   
            
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
            setattr( self, name, items[ assign ] )
        return items[ result ]

    #---------------------------------------------------------------------------
    #  Sets an object trait if a specified option string is found:
    #---------------------------------------------------------------------------
        
    def _option ( self, string, option, name, value ):
        """ Sets a object trait if a specified option string is found.
        """
        col = string.find( option )
        if col >= 0:
            string = string[ : col ] + string[ col + len( option ): ]
            setattr( self, name, value )
        return string

    #---------------------------------------------------------------------------
    #  Parses any of the one character forms of the 'style' trait:
    #---------------------------------------------------------------------------
    
    def _parse_style ( self, value ):
        """ Parses any of the one-character forms of the **style** trait.
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
        empty = False
        if match is not None:
            self.label = match.group( 2 ).strip()
            empty      = (self.label == '')
            value      = match.group( 1 ) + match.group( 3 )
        return ( value, empty )
            
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
        """ Returns a "pretty print" version of a specified Item trait value.
        """
        if value == ignore:
            return ''
        return '%s%s%s' % ( prefix, value, suffix )

    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of a single trait:
    #---------------------------------------------------------------------------
                     
    def _repr_option ( self, value, match, result ):
        """ Returns a "pretty print" version of a single trait.
        """
        if value == match:
            return result
        return ''
        
