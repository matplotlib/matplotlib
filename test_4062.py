import setupext
import unittest

class testFreetypeRobustInstallation (unittest.TestCase):
    def setUp(self):
        self.ft = setupext.FreeType()

    def test_check_None(self):
        self.assertEqual(r"version 2.5.5", self.ft.check())
        
    def test_make_extension(self):
        ext = setupext.make_extension('test', [])
        print "test_make_extension ".ljust(60, '*')
        print ext.include_dirs

    def tearDown(self):
        del self.ft

if "__main__" == __name__:
    unittest.main()
