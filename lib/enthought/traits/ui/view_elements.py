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
#  Symbols defined: ViewElements
#
#------------------------------------------------------------------------------
""" Define the ViewElements class, which is used to define a (typically
class-based) hierarchical name space of related ViewElement objects.

Normally there is a ViewElements object associated with each Traits-based 
class, which contains all of the ViewElement objects associated with the class.
The ViewElements object is also linked to the ViewElements objects of its 
associated class's parent classes.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Trait, HasStrictTraits, List, Dict, Str, Int, Any
    
from enthought.traits.trait_base \
    import enumerate

from view_element \
    import ViewElement

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Trait for contents of a ViewElements object
content_trait = Dict( str, ViewElement )

#-------------------------------------------------------------------------------
#  'ViewElements' class:
#-------------------------------------------------------------------------------

class ViewElements ( HasStrictTraits ):
    """ Defines a hierarchical name space of related ViewElement objects.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # Dictionary containing the named ViewElement items
    content = content_trait 

    #---------------------------------------------------------------------------
    #  Finds a specified ViewElement within the specified (optional) search 
    #  context:
    #---------------------------------------------------------------------------
    
    def find ( self, name, stack = None ):
        """ Finds a specified ViewElement within the specified (optional) search
            context.
        """
        # Assume search starts from the beginning the of the search order:
        i = 0
        
        # If a stack was specified, see if there is a matching entry in the 
        # stack already:
        if stack is not None:
            for ssi in stack:
                if name == ssi.id:
                    # Match found, resume search at next ViewElements object
                    # in the search order:
                    i = ssi.context + 1
                    break
                    
        # Search for a matching name starting at the specified ViewElements
        # object in the search order:
        for j, ves in enumerate( self._get_search_order()[i:] ):
            result = ves.content.get( name )
            if result is not None:
                # Match found. If there is a stack, push matching name and
                # ViewElements context onto it:
                if stack is not None:
                    stack[0:0] = [ SearchStackItem( id      = name, 
                                                    context = i + j ) ]
                    
                # Return the ViewElement object that matched the name:
                return result
                
        # Indicate no match was found:
        return None
        
    #---------------------------------------------------------------------------
    #  Returns a sorted list of all names accessible from the ViewElements 
    #  object that are of a specified (ViewElement) type:
    #---------------------------------------------------------------------------
        
    def filter_by ( self, klass = None ):
        """ Returns a sorted list of all names accessible from the ViewElements
            object that are of a specified (ViewElement) type.
        """
        if klass is None:
            import view
            klass = view.View
        result = []
        
        # Add each item in the search order which is of the right class and
        # which is not already in the result list:
        for ves in self._get_search_order():
            for name, ve in ves.content.items():
                if isinstance( ve, klass ) and (name not in result):
                    result.append( name )
                    
        # Sort the resulting list of names:
        result.sort()
        
        # Return the result:
        return result
        
    #---------------------------------------------------------------------------
    #  Handles the 'parents' list being updated:
    #---------------------------------------------------------------------------
        
    def _parents__changed ( self ):
        self._search_order = None
        
    def _parents_items_changed ( self ):
        self._search_order = None
        
    #---------------------------------------------------------------------------
    #  Returns the current search order (computing it if necessary):
    #---------------------------------------------------------------------------
        
    def _get_search_order ( self ):
        if self._search_order is None:
            self._search_order = self._mro()
        return self._search_order
    
    #---------------------------------------------------------------------------
    #  Compute the Python 'C3' algorithm used to determine a class's 'mro'
    #  and apply it to the 'parents' of the ViewElements to determine the
    #  correct search order:
    #---------------------------------------------------------------------------
            
    def _mro ( self ):
        return self._merge( 
                  [ [ self ] ] + 
                  [ parent._get_search_order()[:] for parent in self.parents ] + 
                  [ self.parents[:] ] )

    def _merge ( self, seqs ):
        result = [] 
        while True:
            # Remove any empty sequences from the list:
            seqs = [ seq for seq in seqs if len( seq ) > 0 ]
            if len( seqs ) == 0: 
                return result
                
            # Find merge candidates among the sequence heads:
            for seq in seqs: 
                candidate = seq[0]
                if len( [ s for s in seqs if candidate in s[1:] ] ) == 0:
                    break
            else:
                raise TraitError, "Inconsistent ViewElements hierarchy"
                
            # Add the candidate to the result:
            result.append( candidate )
            
            # Then remove the candidate:
            for seq in seqs: 
                if seq[0] == candidate: 
                    del seq[0]
                    
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the ViewElements object:
    #---------------------------------------------------------------------------
                    
    def __repr__ ( self ):
        """ Returns a "pretty print" version of the ViewElements object.
        """
        return self.content.__repr__()
        
#-------------------------------------------------------------------------------
#  Define forward reference traits:
#-------------------------------------------------------------------------------

ViewElements.add_class_trait( 'parents',       List( ViewElements ) )
ViewElements.add_class_trait( '_search_order', Any )

#-------------------------------------------------------------------------------
#  'SearchStackItem' class:
#-------------------------------------------------------------------------------

class SearchStackItem ( HasStrictTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # Name that was looked up
    id = Str
    
    # Index into the 'mro' list of ViewElements that the ID was found in
    context = Int
    
