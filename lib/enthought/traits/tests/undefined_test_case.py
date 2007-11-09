import unittest

from enthought.traits.api import HasTraits, Str, Undefined, ReadOnly, Float


class Foo(HasTraits):
    name = Str
    original_name = ReadOnly
    
    bar = Str
    baz = Float
    
    def _name_changed ( self ):
        if self.original_name is Undefined:
            self.original_name = self.name
    

class Bar(HasTraits):
    name = Str( Undefined )

class UndefinedTestCase( unittest.TestCase):

    def test_initial_value(self):
        b = Bar()
        self.failUnless( b.name is Undefined )
        return

    def test_name_change(self):
        b = Bar()
        self.failUnless( b.name is Undefined )
        
        b.name = 'first'
        self.failUnlessEqual( b.name, 'first' )
        return
    
    def test_read_only_write_once(self):
        f = Foo()
        
        self.failUnlessEqual( f.name, '' )
        self.failUnless( f.original_name is Undefined )
        
        f.name = 'first'
        self.failUnlessEqual( f.name, 'first' )
        self.failUnlessEqual( f.original_name, 'first' )
        
        f.name = 'second'
        self.failUnlessEqual( f.name, 'second' )
        self.failUnlessEqual( f.original_name, 'first' )
        
        return
    

    def test_read_only_write_once_from_constructor(self):
        f = Foo(name='first')
        
        f.name = 'first'
        self.failUnlessEqual( f.name, 'first' )
        self.failUnlessEqual( f.original_name, 'first' )
        
        f.name = 'second'
        self.failUnlessEqual( f.name, 'second' )
        self.failUnlessEqual( f.original_name, 'first' )
        
        return

    
    
    
### EOF #######################################################################
