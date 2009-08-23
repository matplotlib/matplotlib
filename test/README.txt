========================================================================
         matplotlib test structure
========================================================================

===== How To Use

= Running

Run the 'run-mpl-test.py' script to execute the test harness.  This must
be run with the version of python that you wish to test matplotlib with.
This means that it must have nose installed (and PIL if image comparison
is to be done).  By default this will pick up whatever python is on your
path, so make sure it is the correct one.

- Command-Line Options
In addition to the standard nose command-line options, there are several
specific to the matplotlib test harness.  They are as follows:

    -t TAG, --with-tag=TAG
                        Will only run test cases that have the specified tag.
                        Each test case should have a 'tag' attribute (if a
                        case does not have one, then it is assumed to be an
                        empty list).  The 'tag' attribute is a list of
                        strings, where each value is a representative propery
                        of the test case.  Example tags are 'qt' or 'units'.
                        This can be specified multiple times.
    --without-tag=TAG   This will run those test cases that do not have the
                        specified tags.
    --clean             This will remove all output files and saved results.
                        If this is specified, no other processing will be
                        performed.
    --all               This will runn all test programs regardless of working
                        directory.
    --keep              Keep any generated output files in a directory called
                        'saved-results'.  This directory will be created if it
                        doesn't already exist.  This directory is in the same
                        location as the test case whose results are being
                        saved.
    --keep-failed       This acts just like '--keep' except will only keeps
                        the results from tests that error or fail.
    --make-test=testName
                        Creates a template test case file in the current
                        directory with the name TestFoo.  Where 'Foo' is the
                        provided test name.


- Running Specific Tests
In order to can specify the exact test case you want to run use the
standard nose mechanism.  For example, if you have the following setup:

TestFoo.py
   def test_func():
      ...

   class TestFoo:
      def test_bar( self ):
         ...
      def test_bug( self ):
         ...

Then to test everything in TestFoo.py do the following:
$> run-mpl-test.py TestFoo.py

To run all tests in the test class TestFoo do this:
$> run-mpl-test.py TestFoo.py:TestFoo

To run the specific 'test_bar' methodd do the following:
$> run-mpl-test.py TestFoo.py:TestFoo.test_bar


= Detecting Test Cases

When running the matplotlib test script it will search for all tests
in the current working directory and below (unless '--all' is specified).
This is provided that the current working directory is a sub-directory
of the matplotlib test directory.  In the event that it is not, then the
matplotlib root test directory will be used and all appropriate test cases
will be run.

This will not search outside of the test structure and will not look in
the mplTest module.  This will only search for test cases in the root
test directory and any of its sub-directories.

= Saving Results

When using the keep flag any generated files in the 'output' directory
are copied to the 'saved-results/<classname>' directory, where <classname>
is the name of the unit-test class.  This means that for each test case
within a given test class, all output files should have unique names.

The 'saved-results' directory will always contain the results from the
last test run.  This is considered a volatile directory since running
the test cases without the '--keep' flag will remove any existing
'saved-results' directory.  This is to ensure the integrity of the
saved results, they will always match the last test run.

= Filtering Tests

In the case of filtering via tags, a unit-test cane have multiple tags.
When running the test program if any tags are specified as 'skip' then
this will take precedence over any tags that might say 'process'.  For
example, if a test case has both the 'gui' and 'qt' tag, but the command-
line is specified with the following flags:
   '--with-tag=gui --without-tag=qt'
then the example test case will not be run because it matches the skip
tag.


===== Directory Structure

There are several directories in the matplotlib test structure. The first
directory is the 'mplTest' directory.  This is the matplotlib test module
and contains the various python scripts that the test harness needs to
run.  The remaining directories are as follows and contain the various test
cases for matplotlib.

mplTest
   This directory does not contain any test cases, rather it is the location
   of the matplotlib specific utilities for performing unit tests.

test_artists
   This directory contains tests that focus on the rendering aspects of
   the various artists.  Essentially the artist derived functionality.

test_backends
   This directory contains various tests that focus on making sure the
   various backend targets work.

test_basemap
   This directory contains test cases that excercise the basemap add-on
   module.

test_cxx
   This directoy contains tests that focus on testing the interface of
   the compiled code contained in matplotlib.

test_mathtext
   This directory contains tests that focus on excercising the mathtext
   sub-system.

test_numerix
   This directory contains tests that focus on validating the numerix
   component.

test_plots
   This directory contains tests that validate the various plot funtions.

test_pylab
   This directory has pylab specific test cases.

test_transforms
   This directory has test cases that focus on testing the various
   transformation and projection functions.

test_matplotlib
   This directory has all other test cases.  This contins test that focus
   on making sure that Axis, Axes, Figure, etc are all acting properly.  This
   has test cases that are general to the overall funtionality of matplotlib.


===== Writing Test Cases

= The Test Case

As per the nose implementation, a test case is ultimately any function that
has the phrase 'test' in its name.  The matplotlib cases however are grouped
into directories, by what is being tested, and from there are grouped into
classes (one class per file), by similarity.

It is desireable that all matplotlib tests follow the same structure to
not only facilitate the writing of test cases, but to make things easier
for maintaining them and keeping things uniform.

There is a class 'MplTestCase' provided to be the base class for all matplotlib
test classes.  This class provides some extra functionality in the form of
verification functions and test data management.

= Comparison Functions

There are several methods provided for testing whether or not a particular
test case should fail or succeed.  The following methods are provided by
the base matplotlib test class:

- MplTestCase.checkEq( expected, actual, msg = "" )
   Fail if the values are not equal, with the given message.

- MplTestCase.checkNeq( expected, actual, msg = "" )
   Fail if the values are equal, with the given message.

- MplTestCase.checkClose( expected, actual, relTol=None, absTol=None, msg="" )
   Fail if the floating point values are not close enough, with the given message.
   You can specify a relative tolerance, absolute tolerance, or both.

- MplTestCase.checkImage( filename, tol = 1.0e-3, msg = "" )
   Check to see if the image is similair to the one stored in the baseline
   directory.  filename can be a fully qualified name (via the 'outFile' method),
   or it can be the name of the file (to be passed into the 'outFile' method).
   The default tolerance is typically fine, but might need to be adjusted in some
   cases (see the 'compareImages' function for more details).  Fails with
   the specified message.

Note that several of the tests will perform image comparison for validation
of a specific plot.  Though not 100% accurate it at least flags potential
failures and signals a human to come and take a closer look.  If an image has
changed and after a human deems the change is acceptable, then updating the
baseline image with the appropriate image from the 'saved-results' directory
(when using the '--keep' or '--keep-failed' command-line arguments) will make
the test pass properly.

Image comparison depends on the python imaging library (PIL) being installed.
If PIL is not installed, then any test cases that rely on it will not
pass.  To not run these test cases, then pass the '--without-tag=PIL'
option on the command-line.

= Directories

Input data files for a given test case should be place in a directory
called 'inputs' with the test case that uses it.  A convienence function
is provided with each test class for accessing input files.

For example if a test case has an input file of the name 'inputs.txt'
you can get the path to the file by calling 'self.inFile("inputs.txt")'.
This is to allow for a uniform convention that all test cases can follow.

Output files are handled just like input files with the exception that
they are written to the 'output' directory and the path name can be
had by calling 'self.outFile'.  It is more important to use this mechanism
for getting the pathname for an output file because it allows for the
management of cleaning up and saving generated output files (It also
significantly reduces the probability of typo errors when specifying
where to place the files).

A Third and final directory used by the test cases is the 'baseline'
directory.  This is where data files used for verifying test results
are stored.  The path name can be had by using the 'self.baseFile'
method.

Accessing these directories can be made simple (and reduce the chance of a
typo) via the following MplTestCase methods:

- MplTestCase.inFile( filename )
   Returns the full pathname of filename in the input data directory.

- MplTestCase.outFile( filename )
   Returns the full pathname of filename in the output data directory.

- MplTestCase.baseFile( filename )
   Returns the full pathname of filename in the baseline data directory.

= Units

Located in the mplTest directory is a set of unit classes.  These classes
are provided for testing the various unitized data interfaces that matplotlib
supports (ie unit conversion).  These are used because they provide a very
strict enforcement of unitized data which will test the entire spectrum of how
unitized data might be used (it is not always meaningful to convert to
a float without specific units given).  This allows us to test for cases that
might accidentally be performing operations that really do not make sense
physically for unitized data.

The provided classes are as follows:
- UnitDbl
   UnitDbl is essentially a unitized floating point number.  It has a
   minimal set of supported units (enough for testing purposes).  All
   of the mathematical operation are provided to fully test any behaviour
   that might occur with unitized data.  Remeber that unitized data has
   rules as to how it can be applied to one another (a value of distance
   cannot be added to a value of time).  Thus we need to guard against any
   accidental "default" conversion that will strip away the meaning of the
   data and render it neutered.

- Epoch
   Epoch is different than a UnitDbl of time.  Time is something that can be
   measured where an Epoch is a specific moment in time.  Epochs are typically
   referenced as an offset from some predetermined epoch.  Conceptally an Epoch
   is like saying 'January 1, 2000 at 12:00 UTC'.  It is a specific
   time, but more importantly it is a time with a frame.  In the example
   the frame is 'UTC'.  This class is provided to test the functionality of
   matplotlib's various routines and mechanisms for dealing with datetimes.

- Duration
   A difference of two epochs is a Duration.  The distinction between a
   Duration and a UnitDbl of time is made because an Epoch can have different
   frames (or units).  In the case of our test Epoch class the two allowed
   frames are 'UTC' and 'ET' (Note that these are rough estimates provided for
   testing purposes and should not be used in production code where accuracy
   of time frames is desired).  As such a Duration also has a frame of
   reference and therefore needs to be called out as different that a simple
   measurement of time since a delta-t in one frame may not be the same in another.



Updating after diff
====================

  python run-mpl-test.py --all --keep-failed
  ./consolidate_diff_images.sh
  # check your images, decide which are good
  python movegood.py
