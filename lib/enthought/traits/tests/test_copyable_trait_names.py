import unittest

from enthought.traits.api import HasTraits, \
    Any, Bool, Delegate, Event, Instance, Property, Str

class Foo(HasTraits):
    a = Any
    b = Bool
    s = Str
    i = Instance(HasTraits)
    e = Event
    d = Delegate( 'i' )

    p = Property
    
    def _get_p(self):
        return self._p
    
    def _set_p(self, p):
        self._p = p
    
    # Read Only Property
    p_ro = Property
    
    def _get_p_ro(self):
        return id(self)
    
class TestCopyableTraitNames( unittest.TestCase ):
    """ Validate that copyable_trait_names returns the appropraite result.
    """
    def setUp(self):

        foo = Foo()
        self.names = foo.copyable_trait_names()
        return
    
    def test_events_not_copyable(self):
        self.failIf( 'e' in self.names )
    
    def test_delegate_not_copyable(self):
        self.failIf( 'd' in self.names )

    def test_read_only_property_not_copyable(self):
        self.failIf( 'p_ro' in self.names )

    
    def test_any_copyable(self):
        self.failUnless( 'a' in self.names )

    def test_bool_copyable(self):
        self.failUnless( 'b' in self.names )

    def test_str_copyable(self):
        self.failUnless( 's' in self.names )

    def test_instance_copyable(self):
        self.failUnless( 'i' in self.names )

    def test_property_copyable(self):
        self.failUnless( 'p' in self.names )

### EOF
