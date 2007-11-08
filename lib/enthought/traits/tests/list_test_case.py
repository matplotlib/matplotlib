import unittest

from enthought.traits.api import HasTraits, Instance, List, Str, TraitError

class Foo(HasTraits):
    l = List(Str)

class Bar(HasTraits):
    name = Str

    
class Baz(HasTraits):
    bars = List(Bar)
    
class BazRef(HasTraits):
    bars = List(Bar, copy='ref')

    
class DeepBaz(HasTraits):
    baz = Instance(Baz)
    
class DeepBazBazRef(HasTraits):
    baz = Instance(BazRef)
    
    
class ListTestCase(unittest.TestCase):
    
    def test_initialized(self):
        f = Foo()
        self.failIfEqual( f.l, None )
        self.failUnlessEqual( len(f.l), 0 )
        return
    
    def test_initializer(self):
        f = Foo( l=['a', 'list'])
        self.failIfEqual( f.l, None )
        self.failUnlessEqual( f.l, ['a', 'list'] )
        return

    def test_type_check(self):
        f = Foo()
        f.l.append('string')
        
        self.failUnlessRaises( TraitError, f.l.append, 123.456 )
        return
    
    def test_append(self):
        f = Foo()
        f.l.append('bar')
        self.failUnlessEqual(f.l, ['bar'])
        return

    def test_remove(self):
        f = Foo()
        f.l.append('bar')
        f.l.remove('bar')
        self.failUnlessEqual(f.l, [])
        return
    
    def test_slice(self):
        f = Foo( l=['zero', 'one', 'two', 'three'] )
        self.failUnlessEqual( f.l[0], 'zero')
        self.failUnlessEqual( f.l[:0], [])
        self.failUnlessEqual( f.l[:1], ['zero'])
        self.failUnlessEqual( f.l[0:1], ['zero'])
        self.failUnlessEqual( f.l[1:], ['one','two','three'])
        self.failUnlessEqual( f.l[-1], 'three')
        self.failUnlessEqual( f.l[-2], 'two')
        self.failUnlessEqual( f.l[:-1], ['zero', 'one', 'two'])
        return
    
    def test_retrieve_reference(self):
        f = Foo( l=['initial', 'value'] )
        
        l = f.l
        self.failUnless( l is f.l )
        
        # no copy on change behavior, l is always a reference
        l.append('change')
        self.failUnlessEqual( f.l, ['initial', 'value', 'change'])

        f.l.append('more change')
        self.failUnlessEqual( l, ['initial', 'value', 'change', 'more change'])
        return

    def test_assignment_makes_copy(self):
        f = Foo( l=['initial', 'value'] )
        l = ['new']
        
        f.l = l
        # same content
        self.failUnlessEqual( l, f.l)
        
        # different objects
        self.failIf( l is f.l )
        
        # which means behaviorally...
        l.append('l change')
        self.failIf( 'l change' in f.l )
        
        f.l.append('f.l change')
        self.failIf( 'f.l change' in l )
        
        return
    
    def test_should_not_allow_none(self):
        f = Foo( l=['initial', 'value'] )
        try:
            f.l = None
            self.fail('None assigned to List trait.')
        except TraitError:
            pass
        
    def test_should_allow_none(self):

        import testoob.testing
        testoob.testing.skip("List traits do not obey 'allow_none'")
        
        # This test is currently failing.  List traits do not obey 'allow_none'
        # Maybe the class definition should fail.
        class Bar(HasTraits):
            l = List(Str, allow_none=True)

        b = Bar()
        b.l = None
        self.failUnlessEqual(b.l, None)
        return

    def test_clone(self):
        baz = Baz()
        for name in ['a', 'b', 'c', 'd']:
            baz.bars.append( Bar(name=name) )
        
        # Clone will clone baz, the bars list, and the objects in the list
        baz_copy = baz.clone_traits()
        
        self.failIf( baz_copy is baz)
        self.failIf( baz_copy.bars is baz.bars)
        
        self.failUnlessEqual( len(baz_copy.bars), len(baz.bars) )
        for bar in baz.bars:
            self.failIf( bar in baz_copy.bars )
        
        baz_bar_names = [ bar.name for bar in baz.bars ]
        baz_copy_bar_names = [ bar.name for bar in baz_copy.bars ]
        baz_bar_names.sort()
        baz_copy_bar_names.sort()
        self.failUnlessEqual( baz_copy_bar_names, baz_bar_names )
        
        return
    
    def test_clone_ref(self):
        baz = BazRef()
        for name in ['a', 'b', 'c', 'd']:
            baz.bars.append( Bar(name=name) )
        
        # Clone will clone baz, the bars list, but the objects in the list
        # will not be cloned because the copy metatrait of the List is 'ref'
        baz_copy = baz.clone_traits()
        
        self.failIf( baz_copy is baz)
        self.failIf( baz_copy.bars is baz.bars)

        self.failUnlessEqual( len(baz_copy.bars), len(baz.bars) )
        for bar in baz.bars:
            self.failUnless( bar in baz_copy.bars )
        
        return
    
        
    def test_clone_deep_baz(self):
        baz = Baz()
        for name in ['a', 'b', 'c', 'd']:
            baz.bars.append( Bar(name=name) )

        deep_baz = DeepBaz( baz=baz )
        
        # Clone will clone deep_baz, deep_baz.baz, the bars list, 
        # and the objects in the list
        deep_baz_copy = deep_baz.clone_traits()
        
        self.failIf( deep_baz_copy is deep_baz)
        self.failIf( deep_baz_copy.baz is deep_baz.baz)

        baz_copy = deep_baz_copy.baz
        
        self.failIf( baz_copy is baz)
        self.failIf( baz_copy.bars is baz.bars)
        
        self.failUnlessEqual( len(baz_copy.bars), len(baz.bars) )
        for bar in baz.bars:
            self.failIf( bar in baz_copy.bars )
        
        baz_bar_names = [ bar.name for bar in baz.bars ]
        baz_copy_bar_names = [ bar.name for bar in baz_copy.bars ]
        baz_bar_names.sort()
        baz_copy_bar_names.sort()
        self.failUnlessEqual( baz_copy_bar_names, baz_bar_names )
        
        return


    def test_clone_deep_baz_ref(self):
        baz = BazRef()
        for name in ['a', 'b', 'c', 'd']:
            baz.bars.append( Bar(name=name) )

        deep_baz = DeepBazBazRef( baz=baz )
        
        deep_baz_copy = deep_baz.clone_traits()

        self.failIf( deep_baz_copy is deep_baz)
        self.failIf( deep_baz_copy.baz is deep_baz.baz)

        baz_copy = deep_baz_copy.baz

        self.failIf( baz_copy is baz)
        self.failIf( baz_copy.bars is baz.bars)

        self.failUnlessEqual( len(baz_copy.bars), len(baz.bars) )
        for bar in baz.bars:
            self.failUnless( bar in baz_copy.bars )
        
        return
    
        
### EOF
        
