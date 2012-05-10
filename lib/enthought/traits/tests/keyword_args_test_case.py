from enthought.traits.api import HasTraits, Instance, Int

import unittest

class Bar(HasTraits):
    b = Int(3)

class Foo(HasTraits):
   bar = Instance(Bar)

class KeyWordArgsTest(unittest.TestCase):
   def test_using_kw(self):
      bar = Bar(b=5)
      foo = Foo(bar=bar)
      self.assertEqual(foo.bar.b, 5)

   def test_not_using_kw(self):
      foo = Foo()
      self.assertEqual(foo.bar, None)
