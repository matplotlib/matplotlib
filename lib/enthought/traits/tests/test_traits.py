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
# Author: David C. Morrill Date: 03/20/2003 Description: Unit Test Case for the
# Traits Package
# ------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import unittest

from enthought.traits.api import *

import enthought.traits.standard as standard

#-------------------------------------------------------------------------------
#  Base unit test classes:
#-------------------------------------------------------------------------------
    
class test_base ( unittest.TestCase ):
   
    def assign ( self, value ):
        self.obj.value = value

    def coerce ( self, value ):
        return value
                
    def check_set ( self ):
        obj = self.obj
        try:
            # Make sure the default value is correct:
            msg   = 'default value'
            value = self._default_value
            self.assertEqual( obj.value, value )
           
            # Iterate over all legal values being tested:
            msg = 'legal values' 
            i   = 0
            for value in self._good_values:
                obj.value = value
                self.assertEqual( obj.value, self.coerce( value ) )
                if i < len( self._mapped_values ):
                    self.assertEqual( obj.value_, self._mapped_values[i] )
                    i += 1
                  
            # Iterate over all illegal values being tested:
            msg = 'illegal values'
            for value in self._bad_values:
                self.assertRaises( TraitError, self.assign, value )
        except:
            print 'Failed while testing %s for value: %s (%s) in %s' % ( 
                msg, value, value.__class__.__name__, self.__class__.__name__ )
            raise
    
class test_base2 ( unittest.TestCase ):
   
    def indexed_assign ( self, list, index, value ):
        list[ index ] = value
   
    def indexed_range_assign ( self, list, index1, index2, value ):
        list[ index1: index2 ] = value
   
    def test_set ( self, name, default_value, good_values, bad_values, 
                   actual_values = None, mapped_values = None ):
        obj = self.obj
        try:
            # Make sure the default value is correct:
            msg   = 'default value'
            value = default_value
            self.assertEqual( getattr( obj, name ), value )
            
            # Iterate over all legal values being tested:
            if actual_values is None:
                actual_values = good_values
            msg = 'legal values'
            i   = 0
            for value in good_values:
                setattr( obj, name, value )
                self.assertEqual( getattr( obj, name ), actual_values[i] )
                if mapped_values is not None:
                    self.assertEqual( getattr( obj, name + '_' ),
                                      mapped_values[i] )
                i += 1
                  
            # Iterate over all illegal values being tested:
            msg = 'illegal values'
            for value in bad_values:
                self.assertRaises( TraitError, setattr, obj, name, value )
        except:
            print 'Failed while testing %s for value: %s (%s) in %s' % ( 
                msg, value, value.__class__.__name__, self.__class__.__name__ )
            raise
 
#-------------------------------------------------------------------------------
#  Trait that can have any value:
#-------------------------------------------------------------------------------

class any_value ( HasTraits ):
    
    # Trait definitions:
    value = Any
    
class test_any_value ( test_base ):
    
    obj = any_value()
 
    _default_value = None
    _good_values   = [ 10.0, 'ten', u'ten', [ 10 ], { 'ten': 10 }, ( 10, ), 
                       None, 1j ]
    _mapped_values = []
    _bad_values    = []
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'int' values: 
#-------------------------------------------------------------------------------
   
class coercible_int_value ( HasTraits ):

    # Trait definitions:
    value = CInt( 99 )
   
class int_value ( HasTraits ):

    # Trait definitions:
    value = Int( 99 )
   
class test_coercible_int_value ( test_any_value ):
   
    obj = coercible_int_value()

    _default_value = 99
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1,
                       '10', '-10', u'10', u'-10' ]
    _bad_values    = [ '10L', '-10L', '10.1', '-10.1', u'10L', u'-10L',
                       u'10.1', u'-10.1', 'ten', u'ten', [ 10 ],
                       { 'ten': 10 }, ( 10, ), None, 1j ]
 
    def coerce ( self, value ):
        try:
            return int( value )
        except:
            try:
                return int( float( value ) )
            except:
                return int( long( value ) )
   
class test_int_value ( test_any_value ):
   
    obj = int_value()

    _default_value = 99
    _good_values   = [ 10, -10 ]
    _bad_values    = [ 'ten', u'ten', [ 10 ], { 'ten': 10 }, ( 10, ), None, 1j,
                       10L, -10L, 10.1, -10.1, '10L', '-10L', '10.1', '-10.1',
                       u'10L', u'-10L', u'10.1', u'-10.1', 
                       '10', '-10', u'10', u'-10'  ]
 
    def coerce ( self, value ):
        try:
            return int( value )
        except:
            try:
                return int( float( value ) )
            except:
                return int( long( value ) )
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'long' values: 
#-------------------------------------------------------------------------------
   
class coercible_long_value ( HasTraits ):
    
    # Trait definitions:
    value = CLong( 99L )
   
class long_value ( HasTraits ):

    # Trait definitions:
    value = Long( 99L )
   
class test_coercible_long_value ( test_any_value ):
    
    obj = coercible_long_value()

    _default_value = 99L
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1, 
                       '10', '-10', '10L', '-10L',
                       u'10', u'-10', u'10L', u'-10L' ]
    _bad_values    = [ '10.1', '-10.1', u'10.1', u'-10.1', 'ten', u'ten',
                       [ 10 ], [ 10l ], { 'ten': 10 }, ( 10, ),
                       ( 10L, ), None, 1j ]
 
    def coerce ( self, value ):
        try:
            return long( value )
        except:
            return long( float( value ) )

class test_long_value ( test_any_value ):
    
    obj = long_value()

    _default_value = 99L
    _good_values   = [ 10, -10, 10L, -10L ]
    _bad_values    = [ 'ten', u'ten', [ 10 ], [ 10l ], { 'ten': 10 }, ( 10, ), 
                       ( 10L, ), None, 1j, 10.1, -10.1,
                       '10', '-10', '10L', '-10L', '10.1', '-10.1',
                       u'10', u'-10', u'10L', u'-10L', u'10.1', u'-10.1' ]
 
    def coerce ( self, value ):
        try:
            return long( value )
        except:
            return long( float( value ) )

   
#-------------------------------------------------------------------------------
#  Trait that can only have 'float' values: 
#-------------------------------------------------------------------------------
   
class coercible_float_value ( HasTraits ):
    
    # Trait definitions:
    value = CFloat( 99.0 )
   
class float_value ( HasTraits ):
    
    # Trait definitions:
    value = Float( 99.0 )
   
class test_coercible_float_value ( test_any_value ):
    
    obj = coercible_float_value()

    _default_value = 99.0
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1, 
                       '10', '-10', '10.1', '-10.1',
                       u'10', u'-10', u'10.1', u'-10.1' ]
    _bad_values    = [ '10L', '-10L', u'10L', u'-10L', 'ten', u'ten',
                       [ 10 ], { 'ten': 10 }, ( 10, ), None, 1j ]
 
    def coerce ( self, value ):
        try:
            return float( value )
        except:
            return float( long( value ) )
   
class test_float_value ( test_any_value ):
    
    obj = float_value()

    _default_value = 99.0
    _good_values   = [ 10, -10, 10.1, -10.1 ]
    _bad_values    = [ 10L, -10L, 'ten', u'ten', [ 10 ], 
                       { 'ten': 10 }, ( 10, ), None, 1j,
                       '10', '-10', '10L', '-10L', '10.1', '-10.1',
                       u'10', u'-10', u'10L', u'-10L', u'10.1', u'-10.1' ]
 
    def coerce ( self, value ):
        try:
            return float( value )
        except:
            return float( long( value ) )

#-------------------------------------------------------------------------------
#  Trait that can only have 'complex' (i.e. imaginary) values: 
#-------------------------------------------------------------------------------
   
class imaginary_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 99.0-99.0j )
   
class test_imaginary_value ( test_any_value ):
    
    obj = imaginary_value()

    _default_value = 99.0-99.0j
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1, 
                       '10', '-10', '10.1', '-10.1',
                       10j, 10+10j, 10-10j, 10.1j, 10.1+10.1j, 10.1-10.1j,
                       '10j', '10+10j', '10-10j' ]
    _bad_values    = [ u'10L', u'-10L', 'ten', [ 10 ],
                       { 'ten': 10 }, ( 10, ), None ]
 
    def coerce ( self, value ):
        try:
            return complex( value )
        except:
            return complex( long( value ) )
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'string' values: 
#-------------------------------------------------------------------------------
   
class string_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 'string' )
   
class test_string_value ( test_any_value ):
    
    obj = string_value()

    _default_value = 'string'
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1, 
                       '10', '-10', '10L', '-10L', '10.1', '-10.1', 
                       'string', u'string', 1j, [ 10 ], [ 'ten' ],
                       { 'ten': 10 }, ( 10, ), None ]
    _bad_values    = []
 
    def coerce ( self, value ):
        return str( value )
   
#-------------------------------------------------------------------------------
#  Trait that can only have 'unicode' values: 
#-------------------------------------------------------------------------------
   
class unicode_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( u'unicode' )
   
class test_unicode_value ( test_string_value ):
    
    obj = unicode_value()

    _default_value = u'unicode'
    _good_values   = [ 10, -10, 10L, -10L, 10.1, -10.1, 
                       '10', '-10', '10L', '-10L', '10.1', '-10.1', 
                       '', u'', 'string', u'string', 1j,
                       [ 10 ], [ 'ten' ], [ u'ten' ], { 'ten': 10 }, ( 10, ), 
                       None ]
    _bad_values    = []
 
    def coerce ( self, value ):
        return str( value )
   
#-------------------------------------------------------------------------------
#  Trait that can only have an 'enumerated list' values: 
#-------------------------------------------------------------------------------
   
class enum_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( [ 1, 'one', 2, 'two', 3, 'three', 4.4, u'four.four' ] )
   
class test_enum_value ( test_any_value ):
    
    obj = enum_value()

    _default_value = 1
    _good_values   = [ 1, 'one', 2, 'two', 3, 'three', 4.4, u'four.four' ]
    _bad_values    = [ 0, 'zero', 4, None ]
    
#-------------------------------------------------------------------------------
#  Trait that can only have a 'mapped' values: 
#-------------------------------------------------------------------------------
   
class mapped_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 'one', { 'one': 1, 'two': 2, 'three': 3 } )
   
class test_mapped_value ( test_any_value ):
    
    obj = mapped_value()

    _default_value = 'one'
    _good_values   = [ 'one', 'two', 'three' ]
    _mapped_values = [ 1, 2, 3 ]
    _bad_values    = [ 'four', 1, 2, 3, [ 1 ], ( 1, ), { 1: 1 }, None ]
   
#-------------------------------------------------------------------------------
#  Trait that must be a unique prefix of an enumerated list of values: 
#-------------------------------------------------------------------------------
   
class prefixlist_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 'one', TraitPrefixList( 'one', 'two', 'three' ) )
   
class test_prefixlist_value ( test_any_value ):
    
    obj = prefixlist_value()

    _default_value = 'one'
    _good_values   = [ 'o', 'on', 'one', 'tw', 'two', 'th', 'thr', 'thre', 
                       'three' ]
    _bad_values    = [ 't', 'one ', ' two', 1, None ]

    def coerce ( self, value ):
        return { 'o': 'one', 'on': 'one', 'tw': 'two', 'th': 'three' }[ 
               value[:2] ]
   
#-------------------------------------------------------------------------------
#  Trait that must be a unique prefix of a mapped set of values: 
#-------------------------------------------------------------------------------
   
class prefixmap_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 'one', 
                   TraitPrefixMap( { 'one': 1, 'two': 2, 'three': 3 } ) )
   
class test_prefixmap_value ( test_any_value ):
    
    obj = prefixmap_value()

    _default_value = 'one'
    _good_values   = [ 'o', 'on', 'one', 'tw', 'two', 'th', 'thr', 'thre', 
                       'three' ]
    _mapped_values = [ 1, 1, 1, 2, 2, 3, 3, 3 ]
    _bad_values    = [ 't', 'one ', ' two', 1, None ]

    def coerce ( self, value ):
        return { 'o': 'one', 'on': 'one', 'tw': 'two', 'th': 'three' }[ 
               value[:2] ]
   
#-------------------------------------------------------------------------------
#  Trait that must be within a specified integer range:
#-------------------------------------------------------------------------------
   
class intrange_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 3, TraitRange( 2, 5 ) )
   
class test_intrange_value ( test_any_value ):
    
    obj = intrange_value()

    _default_value = 3
    _good_values   = [ 2, 3, 4, 5 ]
    _bad_values    = [ 0, 1, 6, 0.999, 6.01, 'two', '0.999', '6.01', None ]
 
    def coerce ( self, value ):
        try:
            return int( value )
        except:
            try:
                return int( float( value ) )
            except:
                return int( long( value ) )
    
#-------------------------------------------------------------------------------
#  Trait that must be within a specified long range:
#-------------------------------------------------------------------------------
 
# No longer supported...

#class test_longrange_value ( test_any_value ):
#    
#    # Trait definitions:
#    value = Trait( 3L, TraitRange( 2L, 5L ) )
#
#    _default_value = 3L
#    _good_values   = [ 2, 3, 4, 5, 2L, 3L, 4L, 5L, 2.0, 3.0, 4.0, 5.0, 
#                       '2', '3', '4', '5', '2L', '3L', '4L', '5L',  
#                       '2.0', '3.0', '4.0', '5.0' ]
#    _bad_values    = [ 0, 1, 6, 0L, 1L, 6L, 0.999, 6.01, 'two', '0.999', '6.01', 
#                       None ]
# 
#    def coerce ( self, value ):
#        try:
#            return long( value )
#        except:
#            return long( float( value ) )
   
#-------------------------------------------------------------------------------
#  Trait that must be within a specified float range:
#-------------------------------------------------------------------------------
   
class floatrange_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 3.0, TraitRange( 2.0, 5.0 ) )
   
class test_floatrange_value ( test_any_value ):
    
    obj = floatrange_value()

    _default_value = 3.0
    _good_values   = [ 2.0, 3.0, 4.0, 5.0, 2.001, 4.999 ] 
    _bad_values    = [ 0, 1, 6, 0L, 1L, 6L, 1.999, 6.01, 'two', '0.999', '6.01', 
                       None ]
 
    def coerce ( self, value ):
        try:
            return float( value )
        except:
            return float( long( value ) )
       
#-------------------------------------------------------------------------------
#  Trait that must be an instance of a particular class (or subclass):
#-------------------------------------------------------------------------------

# Old style class version:
class OTraitTest1:                pass
class OTraitTest2( OTraitTest1 ): pass   
class OTraitTest3( OTraitTest2 ): pass
class OBadTraitTest:              pass

otrait_test1 = OTraitTest1()
   
class instance_value_old ( HasTraits ):
    
    # Trait definitions:
    value = Trait( otrait_test1 )
   
class test_instance_value_old ( test_any_value ):
    
    # Trait definitions:
    obj = instance_value_old()

    _default_value = otrait_test1
    _good_values   = [ otrait_test1, OTraitTest1(), OTraitTest2(), 
                       OTraitTest3() ]
    _bad_values    = [ 0, 0L, 0.0, 0j, None, OTraitTest1, OTraitTest2,
                       OBadTraitTest(), 'string', u'string', [ otrait_test1 ], 
                       ( otrait_test1, ), { 'data': otrait_test1 } ]

# New style class version:
class NTraitTest1( object ):      pass
class NTraitTest2( NTraitTest1 ): pass   
class NTraitTest3( NTraitTest2 ): pass
class NBadTraitTest:              pass

ntrait_test1 = NTraitTest1()
   
class instance_value_new ( HasTraits ):
    
    # Trait definitions:
    value = Trait( ntrait_test1 )
   
class test_instance_value_new ( test_any_value ):
    
    obj = instance_value_new()

    _default_value = ntrait_test1
    _good_values   = [ ntrait_test1, NTraitTest1(), NTraitTest2(), 
                       NTraitTest3() ]
    _bad_values    = [ 0, 0L, 0.0, 0j, None, NTraitTest1, NTraitTest2,
                       NBadTraitTest(), 'string', u'string', [ ntrait_test1 ], 
                       ( ntrait_test1, ), { 'data': ntrait_test1 } ]
       
#-------------------------------------------------------------------------------
#  Trait (using a function) that must be an odd integer:
#-------------------------------------------------------------------------------

def odd_integer ( object, name, value ):
    try:
       float( value )
       if (value % 2) == 1:
          return int( value )
    except:
       pass
    raise TraitError
   
class oddint_value ( HasTraits ):
    
    # Trait definitions:
    value = Trait( 99, odd_integer )
   
class test_oddint_value ( test_any_value ):
    
    obj = oddint_value()

    _default_value = 99
    _good_values   = [  1,   3,   5,   7,   9,  999999999, 
                        1L,  3L,  5L,  7L,  9L,  999999999L, 
                        1.0, 3.0, 5.0, 7.0, 9.0, 999999999.0,
                       -1,  -3,  -5,  -7,  -9, -999999999,
                       -1L, -3L, -5L, -7L, -9L, -999999999L,
                        -1.0, -3.0, -5.0, -7.0, -9.0, -999999999.0 ]
    _bad_values    = [ 0, 2, -2, 1j, None, '1', [ 1 ], ( 1, ), { 1: 1 } ]

#-------------------------------------------------------------------------------
#  Trait that has various notifiers attached:
#-------------------------------------------------------------------------------
   
class notify_value ( HasTraits ):

    # Trait definitions:
    value1_count = Trait( 0 )
    value2_count = Trait( 0 )
    
    def anytrait_changed ( self, trait_name, old, new ):
        if trait_name == 'value1':
           self.value1_count += 1
        elif trait_name == 'value2':
           self.value2_count += 1
        
    def value1_changed ( self, old, new ):
        self.value1_count += 1
        
    def value2_changed ( self, old, new ):
        self.value2_count += 1
   
class test_notify_value ( unittest.TestCase ):

    obj = notify_value()
    
    def __init__ ( self, value ):
        unittest.TestCase.__init__( self, value )
   
    def setUp ( self ):
        obj = self.obj
        obj.value1       = 0
        obj.value2       = 0
        obj.value1_count = 0
        obj.value2_count = 0
        
    def tearDown ( self ):
        obj = self.obj
        obj.on_trait_change( self.on_value1_changed, 'value1', remove = True )
        obj.on_trait_change( self.on_value2_changed, 'value2', remove = True )
        obj.on_trait_change( self.on_anytrait_changed,         remove = True )
    
    def on_anytrait_changed ( self, object, trait_name, old, new ):
        if trait_name == 'value1':
           self.obj.value1_count += 1
        elif trait_name == 'value2':
           self.obj.value2_count += 1
        
    def on_value1_changed ( self ):
        self.obj.value1_count += 1
        
    def on_value2_changed ( self ):
        self.obj.value2_count += 1
    
    def check_simple ( self ):
        obj = self.obj
        obj.value1 = 1
        self.assertEqual( obj.value1_count, 2 )
        self.assertEqual( obj.value2_count, 0 )
        obj.value2 = 1
        self.assertEqual( obj.value1_count, 2 )
        self.assertEqual( obj.value2_count, 2 )
    
    def check_complex ( self ):
        obj = self.obj
        obj.on_trait_change( self.on_value1_changed, 'value1' )
        obj.value1 = 1
        self.assertEqual( obj.value1_count, 3 )
        self.assertEqual( obj.value2_count, 0 )
        obj.on_trait_change( self.on_value2_changed, 'value2' )
        obj.value2 = 1
        self.assertEqual( obj.value1_count, 3 )
        self.assertEqual( obj.value2_count, 3 )
        obj.on_trait_change( self.on_anytrait_changed )
        obj.value1 = 2
        self.assertEqual( obj.value1_count, 7 )
        self.assertEqual( obj.value2_count, 3 )
        obj.value1 = 2
        self.assertEqual( obj.value1_count, 7 )
        self.assertEqual( obj.value2_count, 3 )
        obj.value2 = 2
        self.assertEqual( obj.value1_count, 7 )
        self.assertEqual( obj.value2_count, 7 )
        obj.on_trait_change( self.on_value1_changed, 'value1', remove = True )
        obj.value1 = 3
        self.assertEqual( obj.value1_count, 10 )
        self.assertEqual( obj.value2_count, 7 )
        obj.on_trait_change( self.on_value2_changed, 'value2', remove = True )
        obj.value2 = 3
        self.assertEqual( obj.value1_count, 10 )
        self.assertEqual( obj.value2_count, 10 )
        obj.on_trait_change( self.on_anytrait_changed, remove = True )
        obj.value1 = 4
        self.assertEqual( obj.value1_count, 12 )
        self.assertEqual( obj.value2_count, 10 )
        obj.value2 = 4
        self.assertEqual( obj.value1_count, 12 )
        self.assertEqual( obj.value2_count, 12 )
   
#-------------------------------------------------------------------------------
#  Trait that uses delegation:
#-------------------------------------------------------------------------------

class float_value ( HasTraits ):

    # Trait definitions:    
    value = Trait( 99.0 )
   
class delegate_value ( HasTraits ):
    
    # Trait definitions:    
    value    = Delegate( 'delegate' )
    delegate = Trait( float_value() )
   
class delegate_2_value ( delegate_value ):
    
    # Trait definitions:    
    delegate = Trait( delegate_value() )
         
class delegate_3_value ( delegate_value ):
    
    # Trait definitions:    
    delegate = Trait( delegate_2_value() )
   
class test_delegation_value ( unittest.TestCase ):
    
    def do_delegation_test ( self, obj ):
        self.assertEqual( obj.value, 99.0 )
        parent1 = obj.delegate
        parent2 = parent1.delegate
        parent3 = parent2.delegate
        parent3.value = 3.0
        self.assertEqual( obj.value,     3.0 )
        parent2.value = 2.0
        self.assertEqual( obj.value,     2.0 )
        self.assertEqual( parent3.value, 3.0 )
        parent1.value = 1.0
        self.assertEqual( obj.value,     1.0 )
        self.assertEqual( parent2.value, 2.0 )
        self.assertEqual( parent3.value, 3.0 )
        obj.value = 0.0
        self.assertEqual( obj.value,     0.0 )
        self.assertEqual( parent1.value, 1.0 )
        self.assertEqual( parent2.value, 2.0 )
        self.assertEqual( parent3.value, 3.0 )
        del obj.value
        self.assertEqual( obj.value,     1.0 )
        del parent1.value
        self.assertEqual( obj.value,     2.0 )
        self.assertEqual( parent1.value, 2.0 )
        del parent2.value
        self.assertEqual( obj.value,     3.0 )
        self.assertEqual( parent1.value, 3.0 )
        self.assertEqual( parent2.value, 3.0 )
        del parent3.value
        # Uncommenting the following line allows
        # the last assertions to pass. However, this
        # may not be intended behaviour, so keeping
        # the line commented.
        #del parent2.value
        self.assertEqual( obj.value,     99.0 )
        self.assertEqual( parent1.value, 99.0 )
        self.assertEqual( parent2.value, 99.0 )
        self.assertEqual( parent3.value, 99.0 )
 
    def check_normal ( self ):
        self.do_delegation_test( delegate_3_value() )

#-------------------------------------------------------------------------------
#  Complex (i.e. 'composite') Traits tests:
#-------------------------------------------------------------------------------
   
class complex_value ( HasTraits ):
    
    # Trait definitions:    
    num1 = Trait( 1, TraitRange( 1, 5 ), TraitRange( -5, -1 ) )
    num2 = Trait( 1, TraitRange( 1, 5 ),
                     TraitPrefixList( 'one', 'two', 'three', 'four', 'five' ) )
    num3 = Trait( 1, TraitRange( 1, 5 ), 
                     TraitPrefixMap( { 'one':   1, 'two':  2, 'three': 3, 
                                       'four': 4, 'five': 5 } ) ) 
   
class test_complex_value ( test_base2 ):
    
    # Trait definitions:   
    obj = complex_value()
        
    def check_num1 ( self ):
        self.test_set( 'num1', 1,
            [ 1, 2, 3, 4, 5, -1, -2, -3, -4, -5 ],
            [ 0, 6, -6, '0', '6', '-6', 0.0, 6.0, -6.0, [ 1 ], ( 1, ),
              { 1: 1 }, None ],
            [ 1, 2, 3, 4, 5, -1, -2, -3, -4, -5 ] )
              
##     def check_num2 ( self ):
##         self.test_set( 'num2', 1,
##             [ 1, 2, 3, 4, 5,
##               'one', 'two', 'three', 'four', 'five', 'o', 'on', 'tw', 
##               'th', 'thr', 'thre', 'fo', 'fou', 'fi', 'fiv' ],
##             [ 0, 6, '0', '6', 0.0, 6.0, 't', 'f', 'six', [ 1 ], ( 1, ),
##               { 1: 1 }, None ],
##             [ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 'one', 'two', 
##               'three', 'four', 'five', 'one', 'one', 'two', 'three', 'three',
##               'three', 'four', 'four', 'five', 'five' ] )
              
##     def check_num3 ( self ):
##         self.test_set( 'num3', 1,
##             [ 1, 2, 3, 4, 5,  
##               'one', 'two', 'three', 'four', 'five', 'o', 'on', 'tw', 
##               'th', 'thr', 'thre', 'fo', 'fou', 'fi', 'fiv' ],
##             [ 0, 6, '0', '6', 0.0, 6.0, 't', 'f', 'six', [ 1 ], ( 1, ),
##               { 1: 1 }, None ],
##             [ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 'one', 'two', 
##               'three', 'four', 'five', 'one', 'one', 'two', 'three', 'three',
##               'three', 'four', 'four', 'five', 'five' ],
##             [ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
##               1, 1, 2, 3, 3, 3, 4, 4, 5, 5 ] )

#-------------------------------------------------------------------------------
#  Test traits which are lists:
#-------------------------------------------------------------------------------

class list_value ( HasTraits ):
    
    # Trait definitions:    
    list1 = Trait( [ 2 ], TraitList( Trait( [ 1, 2, 3, 4 ] ), 
                          maxlen = 4 ) )
    list2 = Trait( [ 2 ], TraitList( Trait( [ 1, 2, 3, 4 ] ), 
                          minlen = 1, maxlen = 4 ) )

class test_list_value ( test_base2 ):
    
    obj = list_value()
           
    def del_range ( self, list, index1, index2 ):
        del list[ index1: index2 ]
    
    def test_list ( self, list ):
        self.assertEqual( list, [ 2 ] )
        self.assertEqual( len( list ), 1 )
        list.append( 3 )
        self.assertEqual( len( list ), 2 )
        list[1] = 2
        self.assertEqual( list[1], 2 )
        self.assertEqual( len( list ), 2 )
        list[0] = 1
        self.assertEqual( list[0], 1 )
        self.assertEqual( len( list ), 2 )
        self.assertRaises( TraitError, self.indexed_assign, list, 0, 5 )
        self.assertRaises( TraitError, list.append, 5 )
        self.assertRaises( TraitError, list.extend, [ 1, 2, 3 ] )
        list.extend( [ 3, 4 ] )
        self.assertEqual( list, [ 1 ,2, 3, 4 ] )
        self.assertRaises( TraitError, list.append, 1 )
        del list[1]
        self.assertEqual( list, [ 1, 3, 4 ] )
        del list[0]
        self.assertEqual( list, [ 3, 4 ] )
        list[:0] = [ 1, 2 ]
        self.assertEqual( list, [ 1 ,2, 3, 4 ] )
        self.assertRaises( TraitError, 
                   self.indexed_range_assign, list, 0, 0, [ 1 ] )
        del list[0:3]
        self.assertEqual( list, [ 4 ] )                          
        self.assertRaises( TraitError, 
                   self.indexed_range_assign, list, 0, 0, [ 4, 5 ] )
     
    def check_list1 ( self ):
        self.test_list( self.obj.list1 )
    
    def check_list2 ( self ):
        self.test_list( self.obj.list2 )
        self.assertRaises( TraitError, self.del_range, self.obj.list2, 0, 1 )
             
#-------------------------------------------------------------------------------
#  Traits based on the 'standard' traits module:
#-------------------------------------------------------------------------------
   
class standard_value ( HasTraits ):
    
    # Trait definitions:    
    zip5       = standard.zipcode_5_trait
    zip9       = standard.zipcode_9_trait
    states     = standard.us_states_short_trait
    statel     = standard.us_states_long_trait
    all_states = standard.all_us_states_short_trait
    all_statel = standard.all_us_states_long_trait
    months     = standard.month_short_trait
    monthl     = standard.month_long_trait
    days       = standard.day_of_week_short_trait
    dayl       = standard.day_of_week_long_trait
    phones     = standard.phone_short_trait
    phonel     = standard.phone_long_trait
    ssn        = standard.ssn_trait
   
class test_standard_value ( test_base2 ):
    
    obj = standard_value()
           
    def check_zip5 ( self ):
        self.test_set( 'zip5', '99999',
            [ '00000', '99999', '54321' ],
            [ '0000', '999999', '12a45', ' 12345', ' 1234', '12345 ', '1234 ',
              '1 345', '\n1234', '1234.' ] )
    
    def check_zip9 ( self ):
        self.test_set( 'zip9', '99999-9999',
            [ '00000-0000', '99999-9999', '98765-4321', '12345 6789', 
              '123456789' ],
            [ '00000-00000', '9999-99999', '12a45-6789', ' 12345-67.9', 
              ' 12345-6789', '12345-6789 ', '12345-678 ',
              '1234 56789', '123456 789', '1234\n6789' ] )
    
    def check_states ( self ):
        self.test_set( 'states', 'TX',
            [ 'AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
              'HI', 'ID', 'IA', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
              'ME', 'MI', 'MO', 'MN', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
              'NJ', 'NM', 'NY', 'NV', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
              'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY' ],
            [ 'A', 'U', 'AA', 'LI', 'TXS', '', 'VI', 'PR', 'GU', 'AS', 
              'DC', None, 1 ] )
    
    def check_statel ( self ):
        self.test_set( 'statel', 'Texas',
            [ 'Alab', 'Alas', 'Ca', 'F', 'Florida', 'North D', 'North C',
              'Massa', 'Misso', 'Min', 'Vermont', 'Ve', 'Vi' ],
            [ 'Al', '', 'Utha', 'South', 'North ', ' Alabama', 'Okalhoma',
              'Florida ', 'F ', 'F.' ],
            [ 'Alabama', 'Alaska', 'California', 'Florida', 'Florida', 
              'North Dakota', 'North Carolina', 'Massachusetts', 'Missouri', 
              'Minnesota', 'Vermont', 'Vermont', 'Virginia' ] )
 
    def check_all_states ( self ):
        self.test_set( 'all_states', 'TX',
            [ 'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 
              'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 
              'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 
              'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 
              'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VI', 
              'VA', 'WA', 'WV', 'WI', 'WY' ],
            [ 'A', 'U', 'AA', 'LI', 'TXS', '', None, 1 ] )
    
    def check_all_statel ( self ):
        self.test_set( 'all_statel', 'Texas',
            [ 'Alab', 'Alas', 'Ca', 'F', 'Florida', 'North D', 'North C',
              'Massa', 'Misso', 'Min', 'Vermont', 'Ve', 'Gu', 'Virgin I',
              'Di', 'Am' ],
            [ 'Al', '', 'Utha', 'South', 'North ', ' Alabama', 'Okalhoma',
              'Florida ', 'F ', 'F.', 'G', 'Vi' ],
            [ 'Alabama', 'Alaska', 'California', 'Florida', 'Florida', 
              'North Dakota', 'North Carolina', 'Massachusetts', 'Missouri', 
              'Minnesota', 'Vermont', 'Vermont', 'Guam', 'Virgin Islands',
              'District of Columbia', 'American Samoa' ] )
 
    def check_months ( self ):
        self.test_set( 'months', 'Jan',
            [ 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ],
            [ 'J', 'Ja', 'Janu', ' Jan', 'Jan ', 'Dece', 'N.v', '', 1, None ] )
 
    def check_monthl ( self ):
        self.test_set( 'monthl', 'January',
            [ 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December',
              'Ja', 'F', 'Mar', 'Ap', 'Jun', 'Jul', 'Au', 'S', 'O', 'N', 'D' ],
            [ 'J', 'Ma', 'Ju', 'Aprli', '', ' June', 'July ' ],
            [ 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December',
              'January', 'February', 'March', 'April', 'June', 'July', 'August', 
              'September', 'October', 'November', 'December' ] )
 
    def check_days ( self ):
        self.test_set( 'days', 'Sun',
            [ 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat' ],
            [ 'Sunday', 'Su', ' Sun', 'Sun ', 'Wedn', 'wed', 'Sa' ] )
 
    def check_dayl ( self ):
        self.test_set( 'dayl', 'Sunday',
            [ 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
              'Saturday', 'Su', 'M', 'Tu', 'W', 'Th', 'F', 'Sa' ],
            [ 'S', 'T', 'Snuday', ' Sunday', 'Sunday ', 'Sau' ], 
            [ 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
              'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 
              'Thursday', 'Friday', 'Saturday' ] )
 
    def check_phones ( self ):
        self.test_set( 'phones', '555-1212',
            [ '000-0000', '999 9999', '1234567', '444-3232', 1234567 ],
            [ '000-000', '0000-000', '123-56789', '123-45.8', '', '123,4567' ],
            [  '000-0000', '999 9999', '1234567', '444-3232', '1234567' ] )
 
    def check_phonel ( self ):
        self.test_set( 'phonel', '800-555-1212',
            [ '000-000-0000', '999 999 9999', '1234567890',
              '(333)333-6666', '(512) 512-9876', '(234) 543 8790',
              1234567890 ],
            [ '000-000-000', '999-9999-999', '1234-567-8769', '[123] 456-7890',
              '(123 567-8970', '(12) 456-7894', ' 123-456-7890',
              '123-456-7890 ', '123-4.6-6543', '12 3-456-7689', None,
              123456789, 12345678901l, ( 1234567890, ) ],
            [ '000-000-0000', '999 999 9999', '1234567890',
              '(333)333-6666', '(512) 512-9876', '(234) 543 8790',
              '1234567890' ] )
 
    def check_ssn ( self ):
        self.test_set( 'ssn', '000-00-0000',
            [ '000 00 0000', '123-45-6789', '333 44-5555', '123-45 6789',
              '123456789', 123456789 ],
            [ '000 00 000', '000 000 000', '345-67-8912 ', ' 123-45-6789',
              '23-456-4356', '23.45.6785', '2a-45-7342', '23  45 6789',
              '23 - 45 - 6789', 12345678, 12345678901l, ( 123456789, ) ],
            [ '000 00 0000', '123-45-6789', '333 44-5555', '123-45 6789',
              '123456789', '123456789' ] )
             
#-------------------------------------------------------------------------------
#  Test suites:
#-------------------------------------------------------------------------------

def test_suite ( level = 1 ):
    suites = []
    if level > 0:
        suites.extend( [ unittest.makeSuite( x, 'check_' ) for x in [
            test_any_value,             
            test_int_value,             test_coercible_int_value,
            test_long_value,            test_coercible_long_value,
            test_float_value,           test_coercible_float_value,
            test_imaginary_value,       test_string_value, 
            test_unicode_value,         test_enum_value,   
            test_mapped_value,          test_prefixlist_value, 
            test_prefixmap_value,       test_intrange_value,   
            test_floatrange_value,      test_instance_value_old, 
            test_instance_value_new,    test_oddint_value, 
            test_notify_value,          test_delegation_value, 
            test_complex_value,         test_list_value, 
            test_standard_value 
            ] ] )
    total_suite = unittest.TestSuite( suites )
    return total_suite

def test ( level = 10 ):
    all_tests = test_suite( level )
    runner    = unittest.TextTestRunner()
    runner.run( all_tests )
    return runner

#-------------------------------------------------------------------------------
#  Run tests:
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    test()
