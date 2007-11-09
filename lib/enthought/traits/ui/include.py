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
#  Symbols defined: Include
#------------------------------------------------------------------------------
""" Defines the Include class, which is used to represent a substitutable
element within a user interface View.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Str
    
from view_element \
    import ViewSubElement
                    
#-------------------------------------------------------------------------------
#  'Include' class:
#-------------------------------------------------------------------------------

class Include ( ViewSubElement ):
    """ A substitutable user interface element, i.e., a placeholder in a view
    definition.
    
    When a view object constructs an attribute-editing window, any Include
    objects within the view definition are replaced with a group or item
    defined elsewhere in the object's inheritance tree, based on matching of the
    name of the element. If no matching element is found, the Include object
    is ignored.
    
    An Include object can reference a group or item attribute on a parent class
    or on a subclass. For example, the following class contains a view 
    definition that provides for the possibility that a subclass might add 
    "extra" attributes in the middle of the view::
        
        class Person(HasTraits):
            name = Str
            age = Int
            person_view = View('name', Include('extra'), 'age', kind='modal')
            
    If you directly create an instance of Person, and edit its attributes, 
    the Include object is ignored.
    
    The following class extends Person, and defines a group of "extra"
    attributes to add to the view defined on Person::

        class LocatedPerson(Person):
            street = Str
            city = Str
            state = Str
            zip = Int
            extra = Group('street', 'city', 'state', 'zip')
            
    The attribute-editing window for an instance of LocatedPerson displays
    editors for these extra attributes.
    """
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # The name of the substitutable content
    id = Str 
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, id, **traits ):
        """ Initializes the Include object.
        """
        super( ViewSubElement, self ).__init__( **traits )
        self.id = id
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Include:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a "pretty print" version of the Include object.
        """
        return "<%s>" % self.id 
    
