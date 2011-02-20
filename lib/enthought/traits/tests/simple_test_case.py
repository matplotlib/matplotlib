
import unittest
import sys

from simple import Simple

class SimpleTestCase( unittest.TestCase ):
    
    def wontfix_test_assignment(self):
        """ see enthought trac ticket #416
                http://www.enthought.com/enthought/ticket/416
        """
        print '\nFirst import and assignment'
        simple = self._import_other_to_create_simple()
        print 'Second import and assignment'
        simple = self._import_other_to_create_simple()
    
    def _import_other_to_create_simple(self):
        other_mod = self._import( 'enthought.traits.tests.other', object_name='Other')
        other_klass = other_mod.__getattribute__('Other')
        
        simple = Simple(name='simon')
        other = other_klass(name='other')
        simple.other = other
        self.failUnless( simple.other is other )
        
        simple2 = Simple(name='simon')
        other2 = other_klass(name='other')
        simple2.other = other2

        self.failUnless( simple2.other is other2 )
        return simple
        
    def _import(self, module_name, object_name=''):
        """ Import the action prototype for the given module name.
        """
        print 'sys.module.keys ', [ name for name in sys.modules.keys() if name.endswith('other') ]
        if module_name is not None:
            print sys.modules.keys()
            if sys.modules.has_key( module_name ):
                imported_module = sys.modules[module_name]
                print '*** reload,', module_name
                reload( imported_module )
            else:
                print '*** __import__', module_name
                imported_module = __import__(module_name, globals(), locals(), [object_name])

        else:
            imported_module = None
            
        return imported_module
