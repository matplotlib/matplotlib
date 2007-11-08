
import unittest

from enthought.traits.api import HasTraits, \
    Any, Bool, Delegate, Event, Instance, Str


class Shared(HasTraits):    
    s = Str('new instance of Shared')

class Foo(HasTraits):
    s = Str('new instance of Foo')
    shared = Instance(Shared)
    
class Bar(HasTraits):    
    s = Str('new instance of Bar')
    foo = Instance(Foo)
    shared = Instance(Shared)
    
class Baz(HasTraits):
    s = Str('new instance of Baz')
    bar = Instance(Bar)
    shared = Instance(Shared)
    
class TestCopyTraitsBase( unittest.TestCase ):
    """ Validate that copy_traits
    """
    def setUp(self):
        print '\n**TestCopyTraitsBase.setUp ',
        super(TestCopyTraitsBase,self).setUp()
        self.shared = Shared(s='shared')
        self.foo = Foo(shared=self.shared, s='foo')
        self.bar = Bar(shared=self.shared, foo=self.foo, s='bar')
        self.baz = Baz(shared=self.shared, bar=self.bar, s='baz')

        self.shared2 = Shared( s='shared2' )
        self.foo2 = Foo( shared=self.shared2, s='foo2' )
        self.bar2 = Bar( shared=self.shared2, foo=self.foo2, s='bar2' )
        self.baz2 = Baz( shared=self.shared2, bar=self.bar2, s='baz2' )

        return

    def print_copy(self):
        print '\nfoo.copy:', self.foo.base_trait('shared').copy
        print 'bar.copy:', self.bar.base_trait('shared').copy
        print 'baz.copy:', self.baz.base_trait('shared').copy
        print 'foo2.copy:', self.foo2.base_trait('shared').copy
        print 'bar2.copy:', self.bar2.base_trait('shared').copy
        print 'baz2.copy:', self.baz2.base_trait('shared').copy
        
    def set_shared_copy(self, value):
        """ Change the copy style for the 'shared' traits. """
        #self.print_copy()
        self.foo.base_trait('shared').copy = value
        self.bar.base_trait('shared').copy = value
        self.baz.base_trait('shared').copy = value
        
        # copy is metadata and therefore a shared  a class attribute
        # self.foo2.base_trait('shared').copy = value
        # self.bar2.base_trait('shared').copy = value
        # self.baz2.base_trait('shared').copy = value
        #self.print_copy()
        
class TestCopyTraitsSetup( TestCopyTraitsBase ):

    def setUp(self):
        super(TestCopyTraitsSetup,self).setUp()
        print '\nshared', self.shared
        print 'foo', self.foo
        print 'bar', self.bar
        print 'baz', self.baz
        print '\nshared2', self.shared2
        print 'foo2', self.foo2
        print 'bar2', self.bar2
        print 'baz2', self.baz2
        
        return
    
    def test_setup(self):
        self.failUnless( self.foo is self.bar.foo )
        self.failUnless( self.bar is self.baz.bar )
        self.failUnless( self.foo.shared is self.shared )
        self.failUnless( self.bar.shared is self.shared )
        self.failUnless( self.baz.shared is self.shared )

        self.failUnless( self.foo2 is self.bar2.foo )
        self.failUnless( self.bar2 is self.baz2.bar )
        self.failUnless( self.foo2.shared is self.shared2 )
        self.failUnless( self.bar2.shared is self.shared2 )
        self.failUnless( self.baz2.shared is self.shared2 )
        return
    
    
    
class CopyTraitsTests:

    def test_baz2_s(self):
        self.failUnlessEqual(self.baz2.s, 'baz')
        self.failUnlessEqual(self.baz2.s, self.baz.s)

    def test_baz2_bar_s(self):
        self.failUnlessEqual(self.baz2.bar.s, 'bar')
        self.failUnlessEqual(self.baz2.bar.s, self.baz.bar.s)

    def test_baz2_bar_foo_s(self):
        self.failUnlessEqual(self.baz2.bar.foo.s, 'foo')
        self.failUnlessEqual(self.baz2.bar.foo.s, self.baz.bar.foo.s)
        
    def test_baz2_shared_s(self):
        self.failUnlessEqual(self.baz2.shared.s, 'shared')
        self.failUnlessEqual(self.baz2.bar.shared.s, 'shared')
        self.failUnlessEqual(self.baz2.bar.foo.shared.s, 'shared')

    def test_baz2_bar(self):
        # First hand Instance trait is different and 
        # is not the same object as the source.
        
        self.failIf( self.baz2.bar is None)
        self.failIf( self.baz2.bar is self.bar2 )
        self.failIf( self.baz2.bar is self.baz.bar )
    
    def test_baz2_bar_foo(self):
        # Second hand Instance trait is a different object and 
        # is not the same object as the source.

        self.failIf( self.baz2.bar.foo is None)
        self.failIf( self.baz2.bar.foo is self.foo2 )
        self.failIf( self.baz2.bar.foo is self.baz.bar.foo )


class CopyTraitsSharedCopyNoneTests:    
    def test_baz2_shared(self):
        # First hand Instance trait is a different object and 
        # is not the same object as the source.

        self.failIf( self.baz2.shared is None)
        self.failIf( self.baz2.shared is self.shared2)
        self.failIf( self.baz2.shared is self.shared)

    def test_baz2_bar_shared(self):
        # Second hand Instance that was shared is a different object and
        # not the same object as the source and
        # not the same object as the new first hand instance that was the same.
        # I.e. There are now (at least) two copies of one orginal object.
        
        self.failIf( self.baz2.bar.shared is None )
        self.failIf( self.baz2.bar.shared is self.shared2 )
        self.failIf( self.baz2.bar.shared is self.shared )
        self.failIf( self.baz2.bar.shared is self.baz2.shared )
    
    def test_baz2_bar_foo_shared(self):
        # Third hand Instance that was shared is a different object and
        # not the same object as the source and 
        # not the same object as the new first hand instance that was the same.
        # I.e. There are now (at least) two copies of one orginal object.
        
        self.failIf( self.baz2.bar.foo.shared is None )
        self.failIf( self.baz2.bar.foo.shared is self.shared2 )
        self.failIf( self.baz2.bar.foo.shared is self.shared )
        self.failIf( self.baz2.bar.foo.shared is self.baz2.shared )

    def test_baz2_bar_and_foo_shared(self):
        #
        # THE BEHAVIOR DEMONSTRATED BY THIS TEST CASE DOES NOT SEEM TO BE CORRECT.
        #
        # Second and Third hand Instance object that was shared with first hand 
        # instance are the same as each other but 
        # Every reference to the same original object has been replace by
        # a reference to the same copy of the same source object except the
        # first hand reference which is a different copy.
        # I.e. The shared relationship has been fubarred by copy_traits: it's
        # not maintained, but not completely destroyed.
        self.failUnless( self.baz2.bar.shared is self.baz2.bar.foo.shared )
        self.failIf( self.baz2.shared is self.baz2.bar.foo.shared )


class TestCopyTraitsSharedCopyNone( CopyTraitsTests,
                                    CopyTraitsSharedCopyNoneTests ):
    def setUp(self):
        print '\n***TestCopyTraitsSharedCopyNone',
        #super(TestCopyTraitsSharedCopyNone,self).setUp()
        
        # deep is the default value for Instance trait copy
        self.set_shared_copy('deep')
        return
        
class TestCopyTraitsCopyNotSpecified(  TestCopyTraitsBase, TestCopyTraitsSharedCopyNone ):

    def setUp(self):
        print '\n*TestCopyTraitsCopyNotSpecified',
#        super(TestCopyTraitsCopyNotSpecified,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyNone.setUp(self)
        self.baz2.copy_traits( self.baz )
        return

        
class TestCopyTraitsCopyShallow( TestCopyTraitsBase, TestCopyTraitsSharedCopyNone ):
    
    def setUp(self):
        print '\n*TestCopyTraitsCopyShallow',
#        super(TestCopyTraitsCopyShallow,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyNone.setUp(self)
        self.baz2.copy_traits( self.baz, copy='shallow' )
        return

class TestCopyTraitsCopyDeep( TestCopyTraitsBase, TestCopyTraitsSharedCopyNone ):
    
    def setUp(self):
        print '\n*TestCopyTraitsCopyDeep',
#        super(TestCopyTraitsCopyDeep,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyNone.setUp(self)
        self.baz2.copy_traits( self.baz, copy='deep' )
        return
        




class CopyTraitsSharedCopyRefTests:    
    def test_baz2_shared(self):
        # First hand Instance trait is a different object and 
        # is the same object as the source.

        self.failIf( self.baz2.shared is None)
        self.failIf( self.baz2.shared is self.shared2)
        self.failUnless( self.baz2.shared is self.shared)

    def test_baz2_bar_shared(self):
        self.failIf( self.baz2.bar.shared is None )
        self.failIf( self.baz2.bar.shared is self.shared2 )
        self.failUnless( self.baz2.bar.shared is self.shared )
        self.failUnless( self.baz2.bar.shared is self.baz2.shared )
    
    def test_baz2_bar_foo_shared(self):
        self.failIf( self.baz2.bar.foo.shared is None )
        self.failIf( self.baz2.bar.foo.shared is self.shared2 )
        self.failUnless( self.baz2.bar.foo.shared is self.shared )
        self.failUnless( self.baz2.bar.foo.shared is self.baz2.shared )

    def test_baz2_bar_and_foo_shared(self):
        self.failUnless( self.baz2.bar.shared is self.baz2.bar.foo.shared )
        self.failUnless( self.baz2.shared is self.baz2.bar.foo.shared )


class TestCopyTraitsSharedCopyRef( CopyTraitsTests,
                                    CopyTraitsSharedCopyRefTests ):
    def setUp(self):
        print '\n***TestCopyTraitsSharedCopyRef.setUp ',
        #super(TestCopyTraitsSharedCopyRef,self).setUp()
        self.set_shared_copy('ref')
        return
    pass

# The next three tests demostrate that a 'ref' trait is always copied as a
# reference regardless of the copy argument to copy_traits.  That is, shallow
# and deep are indistinguishable.

class TestCopyTraitsCopyNotSpecifiedSharedRef( TestCopyTraitsBase, TestCopyTraitsSharedCopyRef):

    def setUp(self):
        print '\n*TestCopyTraitsCopyNotSpecifiedSharedRef.setUp',
#        super(TestCopyTraitsCopyNotSpecifiedSharedRef,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyRef.setUp(self)
        self.baz2.copy_traits( self.baz )
        return

        
class TestCopyTraitsCopyShallowSharedRef( TestCopyTraitsBase, TestCopyTraitsSharedCopyRef ):
    
    def setUp(self):
        print '\n*TestCopyTraitsCopyShallowSharedRef.setUp',
#        super(TestCopyTraitsCopyShallowSharedRef,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyRef.setUp(self)
        self.baz2.copy_traits( self.baz, copy='shallow' )
        return

class TestCopyTraitsCopyDeepSharedRef( TestCopyTraitsBase, TestCopyTraitsSharedCopyRef ):
    
    def setUp(self):
        print '\n*TestCopyTraitsCopyDeepSharedRef.setUp',
#        super(TestCopyTraitsCopyDeepSharedRef,self).setUp()
        TestCopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyRef.setUp(self)
        self.baz2.copy_traits( self.baz, copy='deep' )
        return
        
        
        
### EOF
        
        