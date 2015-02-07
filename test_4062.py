import setupext
import unittest
from pprint import pprint
reload(setupext)

class testFreetypeRobustInstallation (unittest.TestCase):
    def setUp(self):
        self.ft = setupext.FreeType()

    def test_check(self):
        self.assertEqual(r"version 2.5.5", self.ft.check())

    def test_get_extension(self):
        '''test FreeType get_extension() for returnedn .include_dirs'''
        print "test_get_extension ".ljust(60, '+')
        ext = self.ft.get_extension()
        print "ext.include_dirs ".ljust(30, '+')
        pprint (ext.include_dirs)

    def test_make_extension(self):
        ext = setupext.make_extension('test', [])
        print "test_make_extension ".ljust(60, '*')
        print ext.include_dirs

    def tearDown(self):
        del self.ft

if "__main__" == __name__:
    unittest.main()
