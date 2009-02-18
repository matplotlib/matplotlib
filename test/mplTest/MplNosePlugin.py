#=======================================================================

import os
import sys
import shutil
import os.path
import optparse

import nose.case
from nose.plugins import Plugin

from path_utils import *
import directories as dirs
from MplTestCase import MplTestCase

#=======================================================================

__all__ = [ 'MplNosePlugin' ]

#=======================================================================
def getInstance( test ):
   """Given a nose test case, will return the actual unit test instance.

   We do this with a function call in case the method for getting the
   actual unit test instance needs to change.
   """
   assert isinstance( test, nose.case.Test )

   if isinstance( test.test, nose.case.MethodTestCase ):
      return test.test.inst
   elif isinstance( test.test, nose.case.FunctionTestCase ):
      return test.test.test
   # elif isinstance( test.test, unittest.TestCase ):
   else:
      return test.test


#=======================================================================
class MplNosePlugin( Plugin ):

   enabled = True
   name = "MplNosePlugin"
   score = 0

   KEEP_NONE = 0
   KEEP_FAIL = 1
   KEEP_ALL  = 2

   TEST_ERRORED = -1
   TEST_FAILED = 0
   TEST_PASSED = 1

   #--------------------------------------------------------------------
   # Some 'property' functions
   def getRootDir( self ):
      # The bottom directory of the stack is the root directory.
      return self.dirStack[0]

   def getInputDir( self ):
      return os.path.join( self.currentDir, dirs.inputDirName )

   def getOutputDir( self ):
      return os.path.join( self.currentDir, dirs.outputDirName )

   def getBaselineRootDir( self ):
      return os.path.join( self.currentDir, dirs.baselineDirName )

   def getSaveRootDir( self ):
      return os.path.join( self.currentDir, dirs.saveDirName )

   rootDir = property( getRootDir )
   inputDir = property( getInputDir )
   outputDir = property( getOutputDir )
   baselineRootDir = property( getBaselineRootDir )
   saveRootDir = property( getSaveRootDir )

   def getBaselineDir( self, test ):
      t = getInstance( test )
      return os.path.join( self.baselineRootDir, t.__class__.__name__ )

   def getSaveDir( self, test ):
      t = getInstance( test )
      return os.path.join( self.saveRootDir, t.__class__.__name__ )

   #--------------------------------------------------------------------
   def saveResults( self, test ):
      """Save the output directory for the gived test."""
      saveDir = self.getSaveDir( test )
      if not os.path.exists( saveDir ):
         mkdir( saveDir, recursive = True )

      outDir = getInstance( test ).outputDir

      for fname in walk( outDir ):
         if os.path.isdir( fname ):
            shutil.copytree( fname, saveDir )
         else:
            shutil.copy( fname, saveDir )

   #--------------------------------------------------------------------
   def filterTestItem( self, item ):
      """Return true if you want the main test selector to collect tests from
         this class, false if you don't, and None if you don't care.

      Parameters:	
         item : An instance of the testable item that has a 'tag' attribute.
      """

      reallyWant = False
      reallyDontWant = False

      if hasattr( item, 'tags' ):
         itemTags = item.tags
      else:
         itemTags = []

      for tag in self.skipTags:
         if tag in itemTags:
            reallyDontWant = True
            break

      for tag in self.includeTags:
         if tag in itemTags:
            reallyWant = True
         else:
            reallyDontWant = True
            break

      if self.includeTags and not itemTags:
         reallyDontWant = True

      if reallyDontWant:
         return False
      if reallyWant:
         return True
      
      return None

   #--------------------------------------------------------------------
   def addError( self, test, err ):
      """Called when a test raises an uncaught exception. DO NOT return a value
         unless you want to stop other plugins from seeing that the test has
         raised an error.

      Parameters:	
         test : nose.case.Test
                the test case
         err : 3-tuple
               sys.exc_info() tuple
      """
      self.testResults.append( (test, self.TEST_ERRORED, err) )

   #--------------------------------------------------------------------
   def addFailure( self, test, err ):
      """Called when a test fails. DO NOT return a value unless you want to
          stop other plugins from seeing that the test has failed.

      Parameters:	
         test : nose.case.Test
                the test case
         err : 3-tuple
               sys.exc_info() tuple
      """
      self.testResults.append( (test, self.TEST_FAILED, err) )

   #--------------------------------------------------------------------
   def addSuccess( self, test ):
      """Called when a test passes. DO NOT return a value unless you want to
         stop other plugins from seeing the passing test.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      self.testResults.append( (test, self.TEST_PASSED, None) )

   #--------------------------------------------------------------------
   def afterContext( self ):
      """Called after a context (generally a module) has been lazy-loaded,
         imported, setup, had its tests loaded and executed, and torn down.
      """
      return None

   #--------------------------------------------------------------------
   def afterDirectory( self, path ):
      """Called after all tests have been loaded from directory at path and run.

      Parameters:	
         path : string
                the directory that has finished processing
      """
      # Set the current directory to the previous directory
      self.currentDir = self.dirStack.pop()
      chdir( self.currentDir )
      return None

   #--------------------------------------------------------------------
   def afterImport( self, filename, module ):
      """Called after module is imported from filename. afterImport is called
         even if the import failed.

      Parameters:	
         filename : string
                    The file that was loaded
         module : string
                  The name of the module
      """
      return None

   #--------------------------------------------------------------------
   def afterTest( self, test ):
      """Called after the test has been run and the result recorded
         (after stopTest).

      Parameters:	
         test : nose.case.Test
                the test case
      """
      return None

   #--------------------------------------------------------------------
   def beforeContext( self ):
      """Called before a context (generally a module) is examined. Since the
         context is not yet loaded, plugins don't get to know what the
         context is; so any context operations should use a stack that is
         pushed in beforeContext and popped in afterContext to ensure they
         operate symmetrically.

         beforeContext and afterContext are mainly useful for tracking and
         restoring global state around possible changes from within a
         context, whatever the context may be. If you need to operate on
         contexts themselves, see startContext and stopContext, which are
         passed the context in question, but are called after it has been
         loaded (imported in the module case).
      """
      return None

   #--------------------------------------------------------------------
   def beforeDirectory( self, path ):
      """Called before tests are loaded from directory at path.

      Parameters:	
         path : string
                the directory that is about to be processed
      """
      # Save the cuurent directory and set to the new directory.
      self.dirStack.append( self.currentDir )
      self.currentDir = path
      chdir( self.currentDir )

      # Remove any existing 'saved-results' directory
      #NOTE: We must do this after setting 'self.currentDir'
      rmdir( self.saveRootDir )

      return None

   #--------------------------------------------------------------------
   def beforeImport( self, filename, module ):
      """Called before module is imported from filename.

      Parameters:	
         filename : string
                    The file that will be loaded
         module : string
                  The name of the module found in file
      """
      return None

   #--------------------------------------------------------------------
   def beforeTest( self, test ):
      """Called before the test is run (before startTest).

      Parameters:	
         test : nose.case.Test
                the test case
      """
      return None

   #--------------------------------------------------------------------
   def begin( self ):
      """Called before any tests are collected or run. Use this to perform
         any setup needed before testing begins.
      """
      return None

   #--------------------------------------------------------------------
   def configure( self, options, conf ):
      """Called after the command line has been parsed, with the parsed
         options and the config container. Here, implement any config
         storage or changes to state or operation that are set by command
         line options.

         Do not return a value from this method unless you want to stop all
         other plugins from being configured.
      """
      self.includeTags = [ t for t in options.mpl_process_tags ]
      self.skipTags = [ t for t in options.mpl_skip_tags ]
      self.keepLevel = options.mpl_keep

      self.currentDir = os.getcwd()
      self.dirStack = []

      self.testResults = []

   #--------------------------------------------------------------------
   def describeTest( self, test ):
      """Return a test description. Called by nose.case.Test.shortDescription.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      return None

   #--------------------------------------------------------------------
   def finalize( self, result ):
      """Called after all report output, including output from all plugins,
         has been sent to the stream. Use this to print final test results
         or perform final cleanup. Return None to allow other plugins to
         continue printing, any other value to stop them.

      Note
         When tests are run under a test runner other than
         nose.core.TextTestRunner, for example when tests are run via
         'python setup.py test', this method may be called before the default
         report output is sent.
      """
      return None

   #--------------------------------------------------------------------
   def formatError( self, test, err ):
      """Called in result.addError, before plugin.addError. If you want to
         replace or modify the error tuple, return a new error tuple.

      Parameters:	
         test : nose.case.Test
                the test case
      err : 3-tuple
            sys.exc_info() tuple
      """
      return err

   #--------------------------------------------------------------------
   def formatFailure( self, test, err ):
      """Called in result.addFailure, before plugin.addFailure. If you want to
         replace or modify the error tuple, return a new error tuple. Since
         this method is chainable, you must return the test as well, so you
         you'll return something like:
               return (test, err)

      Parameters:	
         test : nose.case.Test
                the test case
         err : 3-tuple
               sys.exc_info() tuple
      """
      return None

   #--------------------------------------------------------------------
   def handleError( self, test, err ):
      """Called on addError. To handle the error yourself and prevent normal
         error processing, return a true value.

      Parameters:	
         test : nose.case.Test
                the test case
         err : 3-tuple
               sys.exc_info() tuple
      """
      if (self.keepLevel == self.KEEP_FAIL) or (self.keepLevel == self.KEEP_ALL):
         self.saveResults( test )

      return None

   #--------------------------------------------------------------------
   def handleFailure( self, test, err ):
      """Called on addFailure. To handle the failure yourself and prevent
         normal failure processing, return a true value.

      Parameters:	
         test : nose.case.Test
                the test case
         err : 3-tuple
               sys.exc_info() tuple
      """
      if (self.keepLevel == self.KEEP_FAIL) or (self.keepLevel == self.KEEP_ALL):
         self.saveResults( test )

      return None

   #--------------------------------------------------------------------
   def loadTestsFromDir( self, path ):
      """Return iterable of tests from a directory. May be a generator.
         Each item returned must be a runnable unittest.TestCase
         (or subclass) instance or suite instance. Return None if your
         plugin cannot collect any tests from directory.

      Parameters:	
         path : string
                The path to the directory.
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromFile( self, filename ):
      """Return tests in this file. Return None if you are not interested in
         loading any tests, or an iterable if you are and can load some. May
         be a generator. If you are interested in loading tests from the file
         and encounter no errors, but find no tests, yield False or
         return [False].

      Parameters:	
         filename : string
                    The full path to the file or directory.
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromModule( self, module ):
      """Return iterable of tests in a module. May be a generator. Each
         item returned must be a runnable unittest.TestCase (or subclass)
         instance. Return None if your plugin cannot collect any tests
         from module.

      Parameters:	
         module : python module
                  The module object
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromName( self, name, module=None, importPath=None ):
      """Return tests in this file or module. Return None if you are not able
         to load any tests, or an iterable if you are. May be a generator.

      Parameters:	
         name : string
                The test name. May be a file or module name plus a test
                callable. Use split_test_name to split into parts. Or it might
                be some crazy name of your own devising, in which case, do
                whatever you want.
         module : python module
                  Module from which the name is to be loaded
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromNames( self, names, module=None ):
      """Return a tuple of (tests loaded, remaining names). Return None if you
         are not able to load any tests. Multiple plugins may implement
         loadTestsFromNames; the remaining name list from each will be passed
         to the next as input.

      Parameters:	
         names : iterable
                 List of test names.
         module : python module
                  Module from which the names are to be loaded
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromTestCase( self, cls ):
      """Return tests in this test case class. Return None if you are not able
         to load any tests, or an iterable if you are. May be a generator.

      Parameters:	
         cls : class
               The test case class. Must be subclass of unittest.TestCase.
      """
      return None

   #--------------------------------------------------------------------
   def loadTestsFromTestClass( self, cls ):
      """Return tests in this test class. Class will not be a unittest.TestCase
         subclass. Return None if you are not able to load any tests, an
         iterable if you are. May be a generator.

      Parameters:	
         cls : class
               The test class. Must NOT be subclass of unittest.TestCase.
      """
      return None

   #--------------------------------------------------------------------
   def makeTest( self, obj, parent ):
      """Given an object and its parent, return or yield one or more test
         cases. Each test must be a unittest.TestCase (or subclass) instance.
         This is called before default test loading to allow plugins to load
         an alternate test case or cases for an object. May be a generator.

      Parameters:	
         obj : any object
               The object to be made into a test
         parent : class, module or other object
                  The parent of obj (eg, for a method, the class)
      """
      return None

   #--------------------------------------------------------------------
   def options( self, parser, env = os.environ ):
      """Called to allow plugin to register command line options with the parser.

      Do not return a value from this method unless you want to stop all other
      plugins from setting their options.

      NOTE: By default, parser is a Python optparse.OptionParser instance.
      """
      helpMsg = "The following are options specific to the matplotlib test harness"
      group = optparse.OptionGroup( parser, "Matplotlib Options", helpMsg )

      # Options to handle tags
      helpMsg  = "Will only run test cases that have the specified tag.  Each "
      helpMsg += "test case should have a 'tag' attribute (if a case does not h"
      helpMsg += "ave one, then it is assumed to be an empty list).  The 'tag' "
      helpMsg += "attribute is a list of strings, where each value is a "
      helpMsg += "representative propery of the test case.  Example tags are "
      helpMsg += "'qt' or 'units'. This can be specified multiple times."
      group.add_option( '-t', '--with-tag',
                        action = 'append', type = 'string', dest = 'mpl_process_tags',
                        default = [], metavar = 'TAG', help = helpMsg )

      helpMsg  = "This will run those test cases that do not have the specified tags."
      group.add_option( '--without-tag',
                        action = 'append', type = 'string', dest = 'mpl_skip_tags',
                        default = [], metavar = 'TAG', help = helpMsg )


      # Some Miscellaneous options
      helpMsg  = "This will remove all output files, saved results, and .pyc files.  "
      helpMsg += "If this is specified, no other processing will be performed."
      group.add_option( '--clean',
                        action = "store_true", dest = "mpl_clean",
                        default = False, help = helpMsg )

      helpMsg = "This will run all test programs regardless of working directory."
      group.add_option( '--all',
                        action = "store_true", dest = "mpl_all",
                        default = False, help = helpMsg )


      # Options to handle generated data files
      helpMsg  = "Keep any generated output files in a directory called "
      helpMsg += "'saved-results'.  This directory will be created if it "
      helpMsg += "doesn't already exist.  This directory is in the same "
      helpMsg += "location as the test case whose results are being saved."
      group.add_option( '--keep',
                        action = "store_const", dest = "mpl_keep",
                        default = self.KEEP_NONE, const = self.KEEP_ALL, help = helpMsg )

      helpMsg  = "This acts just like '--keep' except will only keeps the results "
      helpMsg += "from tests that error or fail."
      group.add_option( '--keep-failed',
                        action = "store_const", dest = "mpl_keep",
                        default = self.KEEP_NONE, const = self.KEEP_FAIL, help = helpMsg )


      # Options to create a test case file
      helpMsg  = "Creates a template test case file in the current directory "
      helpMsg += "with the name TestFoo.  Where 'Foo' is the provided test name."
      group.add_option( '--make-test',
                        action = 'store', dest = 'mpl_make_test',
                        default = False, metavar = 'testName', help = helpMsg )


      parser.add_option_group( group )

   #--------------------------------------------------------------------
   def prepareTest( self, test ):
      """Called before the test is run by the test runner. Please note the
         article the in the previous sentence: prepareTest is called only once,
         and is passed the test case or test suite that the test runner will
         execute. It is not called for each individual test case. If you return
         a non-None value, that return value will be run as the test. Use this
         hook to wrap or decorate the test with another function. If you need
         to modify or wrap individual test cases, use prepareTestCase instead.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      return None

   #--------------------------------------------------------------------
   def prepareTestCase( self, test ):
      """Prepare or wrap an individual test case. Called before execution of
         the test. The test passed here is a nose.case.Test instance; the case
         to be executed is in the test attribute of the passed case. To modify
         the test to be run, you should return a callable that takes one
         argument (the test result object) -- it is recommended that you do not
         side-effect the nose.case.Test instance you have been passed.

      Keep in mind that when you replace the test callable you are replacing
      the run() method of the test case -- including the exception handling
      and result calls, etc.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      # Save the dir names in the test class instance to make it available
      # to the individual test cases.
      t = getInstance( test )
      t.inputDir = self.inputDir
      t.outputDir = self.outputDir
      t.baselineDir = self.getBaselineDir( test )
      t.workingDir = self.currentDir

      return None

   #--------------------------------------------------------------------
   def prepareTestLoader( self, loader ):
      """Called before tests are loaded. To replace the test loader, return a
         test loader. To allow other plugins to process the test loader,
         return None. Only one plugin may replace the test loader. Only valid
         when using nose.TestProgram.
   
      Parameters:	
         loader : nose.loader.TestLoader or other loader instance
                  the test loader
      """
      return None

   #--------------------------------------------------------------------
   def prepareTestResult( self, result ):
      """Called before the first test is run. To use a different test result
         handler for all tests than the given result, return a test result
         handler. NOTE however that this handler will only be seen by tests,
         that is, inside of the result proxy system. The TestRunner and
         TestProgram -- whether nose's or other -- will continue to see the
         original result handler. For this reason, it is usually better to
         monkeypatch the result (for instance, if you want to handle some
         exceptions in a unique way). Only one plugin may replace the result,
         but many may monkeypatch it. If you want to monkeypatch and stop
         other plugins from doing so, monkeypatch and return the patched result.

      Parameters:	
         result : nose.result.TextTestResult or other result instance
                  the test result
      """
      return None

   #--------------------------------------------------------------------
   def prepareTestRunner( self, runner ):
      """Called before tests are run. To replace the test runner, return a
         test runner. To allow other plugins to process the test runner,
         return None. Only valid when using nose.TestProgram.

      Parameters:	
         runner : nose.core.TextTestRunner or other runner instance
                  the test runner
      """
      return None

   #--------------------------------------------------------------------
   def report( self, stream ):
      """Called after all error output has been printed. Print your plugin's
         report to the provided stream. Return None to allow other plugins to
         print reports, any other value to stop them.

      Parameters:	
         stream : file-like object
                  stream object; send your output here
      """
      return None

   #--------------------------------------------------------------------
   def setOutputStream( self, stream ):
      """Called before test output begins. To direct test output to a new
         stream, return a stream object, which must implement a write(msg)
         method. If you only want to note the stream, not capture or redirect
         it, then return None.

      Parameters:	
         stream : file-like object
                  the original output stream
      """
      return None

   #--------------------------------------------------------------------
   def startContext( self, context ):
      """Called before context setup and the running of tests in the context.
         Note that tests have already been loaded from the context before this call.

      Parameters:	
         context : module, class or other object
                   the context about to be setup. May be a module or class, or
                   any other object that contains tests.
      """
      return None

   #--------------------------------------------------------------------
   def startTest( self, test ):
      """Called before each test is run. DO NOT return a value unless you want
         to stop other plugins from seeing the test start.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      # make sure there is a fresh output directory to use.
      rmdir( self.outputDir )
      mkdir( self.outputDir, recursive = True )

      # sys.stdout.write( "%s\n     %s     \n" % (test.id(), test.shortDescription()) )
      print "%s" % (test.id())
      print "     %s" % (test.shortDescription())

   #--------------------------------------------------------------------
   def stopContext( self, context ):
      """Called after the tests in a context have run and the context has been
         torn down.

      Parameters:	
         context : module, class or other object
                   the context that has just been torn down.
      """
      return None

   #--------------------------------------------------------------------
   def stopTest( self, test ):
      """Called after each test is run. DO NOT return a value unless you want
         to stop other plugins from seeing that the test has stopped.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      assert test == self.testResults[-1][0]

      if self.keepLevel == self.KEEP_ALL:
         self.saveResults( test )

      # KEEP_FAIL is handled by the 'handleError' and 'handleFailed' methods.

      rmdir( self.outputDir )

   #--------------------------------------------------------------------
   def testName( self, test ):
      """Return a short test name. Called by nose.case.Test.__str__.

      Parameters:	
         test : nose.case.Test
                the test case
      """
      return None

   #--------------------------------------------------------------------
   def wantClass( self, cls ):
      """Return true if you want the main test selector to collect tests from
         this class, false if you don't, and None if you don't care.

      Parameters:	
         cls : class
               The class being examined by the selector
      """
      # Filter out classes that do not inherit from MplTestCase
      if not issubclass( cls, MplTestCase ):
         return False

      return self.filterTestItem( cls )

   #--------------------------------------------------------------------
   def wantDirectory( self, dirname ):
      """Return true if you want test collection to descend into this
         directory, false if you do not, and None if you don't care.

      Parameters:	
         dirname : string
                   Full path to directory being examined by the selector
      """
      # Skip the unit-test utility module.
      if dirname == os.path.join( self.rootDir, 'mplTest' ):
         return False

      return None

   #--------------------------------------------------------------------
   def wantFile( self, file ):
      """Return true if you want to collect tests from this file, false if
         you do not and None if you don't care.

      Parameters:	
         file : string
                Full path to file being examined by the selector
      """
      # Skip anything not under the root test directory
      if self.rootDir not in file:
         return False

      return None

   #--------------------------------------------------------------------
   def wantFunction( self, function ):
      """Return true to collect this function as a test, false to prevent it
         from being collected, and None if you don't care.

      Parameters:	
         function : function
                    The function object being examined by the selector
      """
      #TODO: Filter out functions that exist outside of the test-structure
      name = function.__name__.lower()
      if "disabled" in name: return False
      return self.filterTestItem( function )

   #--------------------------------------------------------------------
   def wantMethod( self, method ):
      """Return true to collect this method as a test, false to prevent it
         from being collected, and None if you don't care.

      Parameters:	
         method : unbound method
                  The method object being examined by the selector
      """
      #TODO: Filter out methods that exist outside of the test-structure
      name = method.__name__.lower()
      if "disabled" in name: return False
      return self.filterTestItem( method )

   #--------------------------------------------------------------------
   def wantModule( self, module ):
      """Return true if you want to collection to descend into this module,
         false to prevent the collector from descending into the module, and
         None if you don't care.

      Parameters:	
         module : python module
                  The module object being examined by the selector
      """
      #TODO: Filter out modules that exist outside of the test-structure
      name = module.__name__.lower()
      if "disabled" in name: return False
      return self.filterTestItem( module )


