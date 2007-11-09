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
# Description: <Traits component>
#------------------------------------------------------------------------------
from enthought.traits.api import HasTraits, TraitProperty

class Test ( HasTraits ):
    
    __traits__ = { }
    
    def __value_get ( self ):
        return self.__dict__.get( '_value', 0 )
        
    def __value_set ( self, value ):
        old_value = self.__dict__.get( '_value', 0 )
        if value != old_value:
            self._value = value
            self.trait_property_changed( 'value', old_value, value )
        
    __traits__[ 'value' ] = TraitProperty( __value_get, __value_set )

    
class Test_1 ( Test ):

    def value_changed ( self, value ):
        print 'value_changed:', value
    
        
class Test_2 ( Test ):

    def anytrait_changed ( self, name, value ):
        print 'anytrait_changed for %s: %s' % ( name, value )
        
        
class Test_3 ( Test_2 ):        

    def value_changed ( self, value ):
        print 'value_changed:', value

        
def on_value_changed ( value ):
    print 'on_value_changed:', value
        
def on_anyvalue_changed ( value ):
    print 'on_anyvalue_changed:', value
    
    
Test_1().value = 'test 1'    
Test_2().value = 'test 2'    
Test_3().value = 'test 3'

test_4 = Test()
test_4.on_trait_change( on_value_changed, 'value' )
test_4.value = 'test 4'
    
test_5 = Test()
test_5.on_trait_change( on_anyvalue_changed )
test_5.value = 'test 5'
    
test_6 = Test()
test_6.on_trait_change( on_value_changed, 'value' )
test_6.on_trait_change( on_anyvalue_changed )
test_6.value = 'test 6'
    
test_7 = Test_3()
test_7.on_trait_change( on_value_changed, 'value' )
test_7.on_trait_change( on_anyvalue_changed )
test_7.value = 'test 7'
    
