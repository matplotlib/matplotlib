#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------
import unittest

from numpy import arange, array, concatenate, ndarray, ravel, take, zeros

from enthought.traits.api import *


# Validator for my simple array type.
def validator(object, name, value):
    type_value = type( value )
    if type_value == ndarray:
        shape = value.shape
        if len( shape ) == 1:
            return value
        elif len( shape ) == 2:
            if shape[1] == 2:
                axis = 1
            elif shape[0] == 2:
                axis = 0
            else:
                raise TraitError
            return ravel( take( value, ( 1, ), axis ) )
    else:
        raise TraitError

class Foo( HasTraits ):
    #a = Trait(array([0.0, 1.0]), validator, desc='foo')
    a = Array()
    event_fired = Bool(False)

    def _a_changed(self):
        self.event_fired = True

class ArrayTestCase( unittest.TestCase ):
    """ Test cases for delegated traits. """


    def test_zero_to_one_element(self):
        """ Test that an event fires when an Array trait changes from zero to
        one element.
        """
        
        f = Foo()
        f.a = zeros((2,), float)
        f.event_fired = False
        
        # Change the array.
        f.a = concatenate((f.a, array([100])))

        # Confirm that the static trait handler was invoked.
        self.assertEqual( f.event_fired, True )

        return

#### EOF ######################################################################
        
