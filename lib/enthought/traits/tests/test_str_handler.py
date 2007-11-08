import unittest

from enthought.traits.api import HasTraits, Str, Trait, TraitError, TraitHandler
from enthought.traits.trait_base import strx

# Validation via function
def validator(object, name, value):
    if isinstance(value, basestring):
        # abitrary rule for testing
        if value.find('fail') < 0:
            return value
        else:
            raise TraitError
    else:
        raise TraitError

# Validation via Handler
class MyHandler(TraitHandler):
    def validate ( self, object, name, value ):
        #print 'myvalidate "%s" %s' % (value, type(value))
        try:
            value = strx( value )
            if value.find('fail') < 0:
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

        return

    def info(self):
        msg = "a string not contining the character sequence 'fail'"
        return msg

class Foo(HasTraits):
    s = Trait('', validator)

class Bar(HasTraits):
    s = Trait('', MyHandler() )

class StrHandlerCase(unittest.TestCase):

    def test_validator_function(self):
        f = Foo()
        self.failUnlessEqual( f.s, '' )

        f.s = 'ok'
        self.failUnlessEqual( f.s, 'ok' )

        self.failUnlessRaises( TraitError, setattr, f, 's', 'should fail.')
        self.failUnlessEqual( f.s, 'ok' )

        return

    def test_validator_handler(self):
        b = Bar()
        self.failUnlessEqual(b.s, '')

        b.s = 'ok'
        self.failUnlessEqual(b.s, 'ok')

        self.failUnlessRaises( TraitError, setattr, b, 's', 'should fail.')
        self.failUnlessEqual( b.s, 'ok')

        return

### EOF

