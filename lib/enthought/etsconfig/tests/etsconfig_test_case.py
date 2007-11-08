""" Tests the 'ETSConfig' configuration object. """


# Standard library imports.
import os, time, unittest

# Enthought library imports.
from enthought.etsconfig.api import ETSConfig


class ETSConfigTestCase(unittest.TestCase):
    """ Tests the 'ETSConfig' configuration object. """

    ###########################################################################
    # 'TestCase' interface.
    ###########################################################################

    #### public methods #######################################################

    def setUp(self):
        """
        Prepares the test fixture before each test method is called.

        """

        return

    def tearDown(self):
        """
        Called immediately after each test method has been called.

        """

        return


    ###########################################################################
    # 'ETSConfigTestCase' interface.
    ###########################################################################

    #### public methods #######################################################

    def test_application_data(self):
        """
        application data

        """

        dirname = ETSConfig.application_data

        self.assertEqual(os.path.exists(dirname), True)
        self.assertEqual(os.path.isdir(dirname), True)

        return

    def test_set_application_data(self):
        """
        set application data

        """

        old = ETSConfig.application_data

        ETSConfig.application_data = 'foo'
        self.assertEqual('foo', ETSConfig.application_data)

        ETSConfig.application_data = old
        self.assertEqual(old, ETSConfig.application_data)

        return


    def test_application_data_is_idempotent(self):
        """
        application data is idempotent

        """

        # Just do the previous test again!
        self.test_application_data()

        return


    def test_write_to_application_data_directory(self):
        """
        write to application data directory

        """

        ETSConfig.company = 'Blah'
        dirname = ETSConfig.application_data

        path = os.path.join(dirname, 'dummy.txt')
        data = str(time.time())

        f = file(path, 'w')
        f.write(data)
        f.close()

        self.assertEqual(os.path.exists(path), True)

        f = file(path)
        result = f.read()
        f.close()

        os.remove(path)

        self.assertEqual(data, result)

        return


    def test_default_company(self):
        """
        default company

        """

        self.assertEqual(ETSConfig.company, 'Enthought')

        return


    def test_set_company(self):
        """
        set company

        """

        old = ETSConfig.company

        ETSConfig.company = 'foo'
        self.assertEqual('foo', ETSConfig.company)

        ETSConfig.company = old
        self.assertEqual(old, ETSConfig.company)

        return


    def _test_default_application_home(self):
        """
        application home

        """

        # This test is only valid when run with the 'main' at the end of this
        # file: "python app_dat_locator_test_case.py", in which case the
        # app_name will be the directory this file is in ('tests').
        app_home = ETSConfig.application_home
        (dirname, app_name) = os.path.split(app_home)

        self.assertEqual(dirname, ETSConfig.application_data)
        self.assertEqual(app_name, 'tests')


    def test_user_data(self):
        """
        user data

        """

        dirname = ETSConfig.user_data

        self.assertEqual(os.path.exists(dirname), True)
        self.assertEqual(os.path.isdir(dirname), True)

        return


    def test_set_user_data(self):
        """
        set user data

        """

        old = ETSConfig.user_data

        ETSConfig.user_data = 'foo'
        self.assertEqual('foo', ETSConfig.user_data)

        ETSConfig.user_data = old
        self.assertEqual(old, ETSConfig.user_data)

        return


    def test_user_data_is_idempotent(self):
        """
        user data is idempotent

        """

        # Just do the previous test again!
        self.test_user_data()

        return


    def test_write_to_user_data_directory(self):
        """
        write to user data directory

        """

        ETSConfig.company = 'Blah'
        dirname = ETSConfig.user_data

        path = os.path.join(dirname, 'dummy.txt')
        data = str(time.time())

        f = file(path, 'w')
        f.write(data)
        f.close()

        self.assertEqual(os.path.exists(path), True)

        f = file(path)
        result = f.read()
        f.close()

        os.remove(path)

        self.assertEqual(data, result)

        return


# For running as an individual set of tests.
if __name__ == '__main__':

    # Add the non-default test of application_home...non-default because it must
    # be run using this module as a script to be valid.
    suite = unittest.TestLoader().loadTestsFromTestCase(ETSConfigTestCase)
    suite.addTest(ETSConfigTestCase('_test_default_application_home'))

    unittest.TextTestRunner(verbosity=2).run(suite)


#### EOF ######################################################################
