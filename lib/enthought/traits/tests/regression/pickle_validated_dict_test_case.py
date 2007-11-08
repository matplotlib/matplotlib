from cPickle import dumps, loads
import unittest

from enthought.traits.api import Dict, HasTraits, Int, List

class C(HasTraits):

    # A dict trait containing a list trait
    a = Dict(Int, List(Int))

    # And we must initialize it to something non-trivial
    def __init__(self):
        super(C, self).__init__()
        self.a = {1 : [2,3]}

class PickleValidatedDictTestCase(unittest.TestCase):
    def test(self):

        # And we must unpickle one
        x = dumps(C())
        try:
            loads(x)
        except AttributeError, e:
            self.fail('Unpickling raised an AttributeError: %s' % e)

# Here is a hack to work around a testoob+traits error:
#
#   Traceback (most recent call last):
#     File "/src/enthought/src/lib/enthought/traits/tests/regression/pickle_validated_dict_test_case.py", line 21, in test
#       x = dumps(C())
#   PicklingError: Can't pickle <class 'pickle_validated_dict_test_case.C'>: import of module pickle_validated_dict_test_case failed
#
# We simply force the two classes to be the same.
import pickle_validated_dict_test_case
pickle_validated_dict_test_case.C = C

if __name__ == '__main__':
    import sys
    unittest.main(argv=sys.argv)
