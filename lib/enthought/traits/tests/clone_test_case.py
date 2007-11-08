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
import unittest

from enthought.traits.api import HasTraits, Instance, Str, Any, Property

class Foo(HasTraits):
    s = Str
    
class ClassWithAny(HasTraits):
    x = Property
    _x = Any
    def _get_x(self):
        return self._x
    def _set_x(self,x):
        self._x = x

class ClassWithInstance(HasTraits):
    x = Property
    _x = Instance(Foo)
    def _get_x(self):
        return self._x
    def _set_x(self,x):
        self._x = x
        
class ClassWithClassAttribute(HasTraits):
    name = 'class defined name'
    foo = Str
    

class BazAny(HasTraits):
    other = Any

class BarAny(HasTraits):
    other = Any
    
    
class BazInstance(HasTraits):
    # A BarInstance owned by this object.
    other = Instance('BarInstance')
    
    # A Foo owned by this object and not referenced by others.
    unique = Instance(Foo)
    
    # A Foo owned by this object and referenced by others.
    shared = Instance(Foo)
    
    # A Foo not owned by this object, may or may not be shared with other
    # objects found via owned references (e.g. other.ref). For the tests,
    # ref will always reference a Foo that is not owned by any of the objects
    # reachable via owned references, and therefore, that Foo object should
    # not be cloned.
    ref = Instance(Foo, copy='ref')

class BarInstance(HasTraits):
    # used as circular reference back to owning BazInstance
    other = Instance('BazInstance', copy='ref')
    
    # A Foo owned by this object and not referenced by others.
    unique = Instance(Foo)

    # A Foo owned by the 'other' object and referenced by this object.
    shared = Instance(Foo, copy='ref')
    
    # A Foo not owned by this object, may or may not be shared with other
    # objects found via owned references (e.g. other.ref). For the tests,
    # ref will always reference a Foo that is not owned by any of the objects
    # reachable via owned references, and therefore, that Foo object should
    # not be cloned.
    ref = Instance(Foo, copy='ref')

    
    
class CloneTestCase( unittest.TestCase ) :
    """ Test cases for traits clone """
    
    def test_any(self) :
        b = ClassWithAny()

        f = Foo()
        f.s = 'the f'
        
        b.x = f
        
        bc = b.clone_traits( traits='all', copy='deep')
        self.assertNotEqual( id(bc.x), id(f), 'Foo x not cloned')
        
        return
    
    def test_instance(self) :
        b = ClassWithInstance()

        f = Foo()
        f.s = 'the f'
        
        b.x = f
        
        bc = b.clone_traits(traits='all', copy='deep')
        self.assertNotEqual( id(bc.x), id(f), 'Foo x not cloned')
        
        return
        
    def test_class_attribute_missing(self):
        """ This test demonstrates a problem with Traits objects with class
        attributes.  A change to the value of a class attribute via one
        instance causes the attribute to be removed from other instances.
        
        AttributeError: 'ClassWithClassAttribute' object has no attribute 'name'
        """
        
        s = 'class defined name'

        c = ClassWithClassAttribute()
        
        self.assertEqual( s, c.name )

        c2 = ClassWithClassAttribute()
        self.assertEqual( s, c.name )
        self.assertEqual( s, c2.name )
        
        s2 = 'name class attribute changed via clone'
        c2.name = s2
        self.assertEqual( s2, c2.name )
        
        # this is failing with 
        # AttributeError: 'ClassWithClassAttribute' object has no attribute 'name'
        self.assertEqual( s, c.name )
        
        return

    def test_Any_circular_references(self):
        
        # Demonstrates that Any traits default to copy='ref'
        bar = BarAny()
        
        baz = BazAny()
        
        bar.other = baz
        baz.other = bar
        
        bar_copy = bar.clone_traits()
        
        self.failIf( bar_copy is bar )
        self.failUnless( bar_copy.other is baz)
        self.failUnless( bar_copy.other.other is bar)


    def test_Any_circular_references_deep(self):
        
        # Demonstrates that Any traits can be forced to deep copy.
        bar = BarAny()
        baz = BazAny()
        bar.other = baz
        baz.other = bar
        
        bar_copy = bar.clone_traits(copy='deep')
        
        self.failIf( bar_copy is bar )
        self.failIf( bar_copy.other is baz)
        self.failIf( bar_copy.other.other is bar)
        self.failUnless( bar_copy.other.other is bar_copy)

    def test_Instance_circular_references(self):

        ref = Foo(s='ref')
        bar_unique = Foo(s='bar.foo')
        shared = Foo(s='shared')
        baz_unique = Foo(s='baz.unique')
        
        baz = BazInstance()
        baz.unique = baz_unique
        baz.shared = shared
        baz.ref = ref

        bar = BarInstance()
        bar.unique = bar_unique
        bar.shared = shared
        bar.ref = ref

        bar.other = baz
        baz.other = bar
        
        baz_copy = baz.clone_traits()

        # Check Baz and Baz attributes....
        self.failIf( baz_copy is baz )
        self.failIf( baz_copy.other is bar)
        self.failIf( baz_copy.unique is baz.unique )
        self.failIf( baz_copy.shared is baz.shared )
        self.failUnless( baz_copy.ref is ref )

        # Check Bar and Bar attributes....
        bar_copy = baz_copy.other

        # Check the Bar owned object
        self.failIf( bar_copy.unique is bar.unique )
        
        # Check the Bar reference to an object 'outside' the cloned graph.
        self.failUnless( bar_copy.ref is ref )

        # Check references to objects that where cloned, they should reference
        # the new clones not the original objects.
        # FROM THIS POINT DOWN ALL TESTS FAIL
        self.failIf( bar_copy.other is baz)
        self.failUnless( bar_copy.other is baz_copy)
        self.failIf( bar_copy.shared is bar.shared )
        self.failUnless( bar_copy.shared is baz_copy.shared )

        

    def test_Instance_circular_references_deep(self):
        
        ref = Foo(s='ref')
        bar_unique = Foo(s='bar.foo')
        shared = Foo(s='shared')
        baz_unique = Foo(s='baz.unique')
        
        baz = BazInstance()
        baz.unique = baz_unique
        baz.shared = shared
        baz.ref = ref

        bar = BarInstance()
        bar.unique = bar_unique
        bar.shared = shared
        bar.ref = ref

        bar.other = baz
        baz.other = bar
        
        baz_copy = baz.clone_traits(copy='deep')

        # Check Baz and Baz attributes....
        self.failIf( baz_copy is baz )
        self.failIf( baz_copy.other is bar)
        self.failIf( baz_copy.unique is baz.unique )
        self.failIf( baz_copy.shared is baz.shared )
        # baz_copy.ref is checked below with bar_copy.ref.
        
        
        # Check Bar and Bar attributes....
        bar_copy = baz_copy.other

        # Chedk the Bar owned object
        self.failIf( bar_copy.unique is bar.unique )
        
        # Since the two original 'ref' links were to a shared object, 
        # the cloned links should be to a shared object.
        self.failUnless( baz_copy.ref is bar_copy.ref )
        
        # FROM THIS POINT DOWN ALL TESTS FAIL
        # But not the original object.
        # Expect deep to copy objects linked via copy='ref'.
        self.failIf( bar_copy.ref is ref )

        # Check references to objects that where cloned, they should reference
        # the new clones not the original objects.
        # NOTE: all of these fail.
        self.failIf( bar_copy.other is baz)
        self.failUnless( bar_copy.other is baz_copy)
        self.failIf( bar_copy.shared is bar.shared )
        self.failUnless( bar_copy.shared is baz_copy.shared )



#
# support running this test individually, from the command-line as a script
#
if __name__ == '__main__':
    unittest.main()
    
#### EOF ######################################################################
