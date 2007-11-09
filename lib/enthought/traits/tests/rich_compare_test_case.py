import unittest

from numpy import array, alltrue

from enthought.traits.api import HasTraits, Any, Array, Str

class IdentityCompare(HasTraits):
    bar = Any(rich_compare=False)

class RichCompare(HasTraits):
    bar = Any(rich_compare=True)

class RichCompareTests:    
    
    def bar_changed(self, object, trait, old, new):
        self.changed_object = object
        self.changed_trait = trait
        self.changed_old = old
        self.changed_new = new
        self.changed_count += 1
    
    def reset_change_tracker(self):
        self.changed_object = None
        self.changed_trait = None
        self.changed_old = None
        self.changed_new = None
        self.changed_count = 0
        
    def check_tracker(self, object, trait, old, new, count):
        self.failUnlessEqual( count, self.changed_count )
        self.failUnless( object is self.changed_object )
        self.failUnlessEqual( trait, self.changed_trait )
        self.failUnless( old is self.changed_old )
        self.failUnless( new is self.changed_new )
        return
    
    def test_id_first_assignment(self):
        ic = IdentityCompare()
        ic.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()
        
        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker( ic, 'bar', default_value, self.a, 1 )
        return
    
    def test_rich_first_assignment(self):
        rich = RichCompare()
        rich.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )
        return

    def test_id_same_object(self):
        ic = IdentityCompare()
        ic.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker( ic, 'bar', default_value, self.a, 1 )
    
        ic.bar = self.a
        self.check_tracker( ic, 'bar', default_value, self.a, 1 )
        
        return

    def test_rich_same_object(self):
        rich = RichCompare()
        rich.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )

        rich.bar = self.a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )
        return

    def test_id_different_object(self):
        ic = IdentityCompare()
        ic.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker( ic, 'bar', default_value, self.a, 1 )
    
        ic.bar = self.different_from_a
        self.check_tracker( ic, 'bar', self.a, self.different_from_a, 2 )
        
        return

    def test_rich_different_object(self):
        rich = RichCompare()
        rich.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )

        rich.bar = self.different_from_a
        self.check_tracker( rich, 'bar', self.a, self.different_from_a, 2 )
        return
    
    def test_id_different_object_same_as(self):
        ic = IdentityCompare()
        ic.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker( ic, 'bar', default_value, self.a, 1 )
    
        ic.bar = self.same_as_a
        self.check_tracker( ic, 'bar', self.a, self.same_as_a, 2 )
        
        return

    def test_rich_different_object_same_as(self):
        rich = RichCompare()
        rich.on_trait_change( self.bar_changed, 'bar' )
        
        self.reset_change_tracker()

        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )

        # Values of a and same_as_a are the same and should therefore not
        # be considered a change.
        rich.bar = self.same_as_a
        self.check_tracker( rich, 'bar', default_value, self.a, 1 )
        return
    

class RichCompareArrayTestCase(unittest.TestCase, RichCompareTests):
    
    def setUp(self):
        self.a = array([1,2,3])
        self.same_as_a = array([1,2,3])
        self.different_from_a = array([3,2,1])
        return

    def test_assumptions(self):
        self.failIf( self.a is self.same_as_a )
        self.failIf( self.a is self.different_from_a )

        self.failUnless( alltrue( self.a == self.same_as_a ) )
        self.failIf( alltrue( self.a == self.different_from_a ) )
        return
    

class Foo(HasTraits):
    name = Str
    
    def __ne__(self, other):
        # Traits uses != to do the rich compare.  The default implementation
        # of __ne__ is to compare the object identities.
        return self.name != other.name

    def __eq__(self, other):
        # Not required, but a good idea to make __eq__ and __ne__ compatible
        return self.name == other.name
    
class RichCompareHasTraitsTestCase(unittest.TestCase, RichCompareTests):
    
    def setUp(self):
        self.a = Foo(name='a')
        self.same_as_a = Foo(name='a')
        self.different_from_a = Foo(name='not a')
        
#        print '\na'
#        self.a.print_traits()
#        print '\nsame_as_a'
#        self.same_as_a.print_traits()
#        print '\ndifferent_from_a'
#        self.different_from_a.print_traits()

        return

    def test_assumptions(self):
        self.failIf( self.a is self.same_as_a )
        self.failIf( self.a is self.different_from_a )

        self.failUnless( self.a.name == self.same_as_a.name )
        self.failIf( self.a.name == self.different_from_a.name )
        return
    

### EOF

