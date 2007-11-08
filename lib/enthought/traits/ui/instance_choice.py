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
# Date:   08/25/2005
#
#  Symbols defined: InstanceChoiceItem
#                   InstanceChoice
#                   InstanceFactoryChoice
#                   InstanceDropChoice
#
#------------------------------------------------------------------------------
""" Defines the various instance descriptors used by the instance editor and
instance editor factory classes.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import HasPrivateTraits, Str, Any, Dict, Tuple, Callable, true, false

from ui_traits \
    import AView

from helper \
    import user_name_for

#-------------------------------------------------------------------------------
#  'InstanceChoiceItem' class:
#-------------------------------------------------------------------------------

class InstanceChoiceItem ( HasPrivateTraits ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # User interface name for the item
    name = Str

    # View associated with this item
    view = AView

    # Does this item create new instances?
    is_factory = false

    #---------------------------------------------------------------------------
    #  Returns the name of the item:
    #---------------------------------------------------------------------------

    def get_name ( self, object = None ):
        """ Returns the name of the item.
        """
        return self.name

    #---------------------------------------------------------------------------
    #  Return the view associated with the object:
    #---------------------------------------------------------------------------

    def get_view ( self ):
        """ Returns the view associated with the object.
        """
        return self.view

    #---------------------------------------------------------------------------
    #  Returns the object associated with the item:
    #---------------------------------------------------------------------------

    def get_object ( self ):
        """ Returns the object associated with the item.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Indicates whether a specified object is compatible with the item:
    #---------------------------------------------------------------------------

    def is_compatible ( self, object ):
        """ Indicates whether a specified object is compatible with the item.
        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    #  Indicates whether the item can be selected by the user:
    #---------------------------------------------------------------------------

    def is_selectable ( self ):
        """ Indicates whether the item can be selected by the user.
        """
        return True

    #---------------------------------------------------------------------------
    #  Indicates whether the item supports drag and drop:
    #---------------------------------------------------------------------------

    def is_droppable ( self ):
        """ Indicates whether the item supports drag and drop.
        """
        return False

#-------------------------------------------------------------------------------
#  'InstanceChoice' class:
#-------------------------------------------------------------------------------

class InstanceChoice ( InstanceChoiceItem ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Object associated with the item
    object = Any

    #---------------------------------------------------------------------------
    #  Returns the name of the item:
    #---------------------------------------------------------------------------

    def get_name ( self, object = None ):
        """ Returns the name of the item.
        """
        if self.name != '':
            return self.name

        name = getattr( self.object, 'name', None )
        if isinstance(name, basestring):
            return name

        return user_name_for( self.object.__class__.__name__ )

    #---------------------------------------------------------------------------
    #  Returns the object associated with the item:
    #---------------------------------------------------------------------------

    def get_object ( self ):
        """ Returns the object associated with the item.
        """
        return self.object

    #---------------------------------------------------------------------------
    #  Indicates whether a specified object is compatible with the item:
    #---------------------------------------------------------------------------

    def is_compatible ( self, object ):
        """ Indicates whether a specified object is compatible with the item.
        """
        return (object is self.object)

#-------------------------------------------------------------------------------
#  'InstanceFactoryChoice' class:
#-------------------------------------------------------------------------------

class InstanceFactoryChoice ( InstanceChoiceItem ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Indicates whether an instance compatible with this item can be dragged and
    # dropped rather than created
    droppable = false

    # Indicates whether the item can be selected by the user
    selectable = true

    # A class (or other callable) that can be used to create an item compatible
    # with this item
    klass = Callable

    # Tuple of arguments to pass to **klass** to create an instance
    args = Tuple

    # Dictionary of arguments to pass to **klass** to create an instance
    kw_args = Dict( Str, Any )

    # Does this item create new instances? This value overrides the default.
    is_factory = True

    #---------------------------------------------------------------------------
    #  Returns the name of the item:
    #---------------------------------------------------------------------------

    def get_name ( self, object = None ):
        """ Returns the name of the item.
        """
        if self.name != '':
            return self.name

        name = getattr( object, 'name', None )
        if isinstance(name, basestring):
            return name

        if issubclass( type( self.klass ), type ):
            klass = self.klass
        else:
            klass = self.get_object().__class__

        return user_name_for( klass.__name__ )

    #---------------------------------------------------------------------------
    #  Returns the object associated with the item:
    #---------------------------------------------------------------------------

    def get_object ( self ):
        """ Returns the object associated with the item.
        """
        return self.klass( *self.args, **self.kw_args )

    #---------------------------------------------------------------------------
    #  Indicates whether the item supports drag and drop:
    #---------------------------------------------------------------------------

    def is_droppable ( self ):
        """ Indicates whether the item supports drag and drop.
        """
        return self.droppable

    #---------------------------------------------------------------------------
    #  Indicates whether a specified object is compatible with the item:
    #---------------------------------------------------------------------------

    def is_compatible ( self, object ):
        """ Indicates whether a specified object is compatible with the item.
        """
        if issubclass( type( self.klass ), type ):
            return isinstance( object, self.klass )
        return isinstance( object, self.get_object().__class__ )

    #---------------------------------------------------------------------------
    #  Indicates whether the item can be selected by the user:
    #---------------------------------------------------------------------------

    def is_selectable ( self ):
        """ Indicates whether the item can be selected by the user.
        """
        return self.selectable

#-------------------------------------------------------------------------------
#  'InstanceDropChoice' class:
#-------------------------------------------------------------------------------

class InstanceDropChoice ( InstanceFactoryChoice ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Indicates whether an instance compatible with this item can be dragged and
    # dropped rather than created . This value overrides the default.
    droppable = True

    # Indicates whether the item can be selected by the user. This value 
    # overrides the default.
    selectable = False

    # Does this item create new instances? This value overrides the default.
    is_factory = False

