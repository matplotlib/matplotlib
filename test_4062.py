import setupext
import unittest

class testFreetypeRobustInstallation (unittest.TestCase):
    def setUp(self):
        self.ft = setupext.FreeType()

    def tearDown(self):
        del self.ft
