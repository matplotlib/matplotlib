#!/bin/env python
"""

   TODO-NOTES:
   Command-line executable that runs the tests.
      -- nice report on test pass/fail status
      -- hooks to add coverage checking and reporting

   Utilities
      -- image comparison tools (non-PIL dependant)

"""
import os
import sys
import os.path

# Save stdout/stderr
originalStdout = sys.stdout
originalStderr = sys.stderr

# get the current directory and the root test directory
cwd = os.path.abspath( os.getcwd() )
root = os.path.dirname( os.path.abspath( sys.argv[0] ) )
sys.path = [ root ] + sys.path

# command-line arguments
args = [ arg for arg in sys.argv ]

# determine the actual working directory to use
if root in cwd:
   working = cwd
else:
   working = root

if '--all' in args:
   working = root

# print "DBG: mpl.test.run - cwd = '%s'" % (cwd)
# print "DBG: mpl.test.run - root = '%s'" % (root)
# print "DBG: mpl.test.run - working = '%s'" % (working)

# make the working directory current
os.chdir( working )

import nose
import nose.plugins.builtin
import matplotlib
from matplotlib.testing.noseclasses import KnownFailure

for arg in args:
   # We need to do this here, because we do not actually want nose to start.
   if arg.startswith( '--make-test=' ):
      testname = arg[ 12: ]
      # Remove any surrounding quotation marks
      if (testname[0] == '"' and testname[-1] == '"') or \
         (testname[0] == "'" and testname[-1] == "'"):
         testname = testname[1:-1]

      filename = os.path.join( cwd, 'Test' + testname + '.py' )
      templName = os.path.join( root, 'mplTest', "TestTEMPLATE.py" )

      fin = open( templName, "r" )
      fout = open( filename, "w" )

      lines = fin.readlines()
      for line in lines:
         newline = line.replace( 'UNITTEST', testname )
         fout.write( newline )

      fin.close()
      fout.close()

      print "Generated '%s'" % (filename)

      sys.exit( 0 )

### Run nose
args.append('.')
args.extend( matplotlib.default_test_modules )

plugins =  [plugin() for plugin in nose.plugins.builtin.plugins]
plugins.extend( [ KnownFailure() ])

success = nose.run( argv = args,
                    plugins = plugins,
                    )

### do other stuff here


# $> nosetests [-w <working_directory>]
# Run a specific test
#    $> nosetests tests/test_stuff.py:test_function
#    $> nosetests tests/test_stuff.py:TestClass.test_method

# Restore the original stdout/stderr
sys.stdout = originalStdout
sys.stderr = originalStderr

sys.exit(not success)
