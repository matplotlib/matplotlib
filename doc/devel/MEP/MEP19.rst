===============================
 MEP19: Continuous Integration
===============================

Status
======

**Completed**

Branches and Pull requests
==========================

Abstract
========

matplotlib could benefit from better and more reliable continuous
integration, both for testing and building installers and
documentation.

Detailed description
====================

Current state-of-the-art
------------------------

**Testing**

matplotlib currently uses Travis-CI for automated tests.  While
Travis-CI should be praised for how much it does as a free service, it
has a number of shortcomings:

- It often fails due to network timeouts when installing dependencies.

- It often fails for inexplicable reasons.

- build or test products can only be saved from build off of branches
  on the main repo, not pull requests, so it is often difficult to
  "post mortem" analyse what went wrong.  This is particularly
  frustrating when the failure can not be subsequently reproduced
  locally.

- It is not extremely fast.  matplotlib's cpu and memory requirements
  for testing are much higher than the average Python project.

- It only tests on Ubuntu Linux, and we have only minimal control over
  the specifics of the platform.  It can be upgraded at any time
  outside of our control, causing unexpected delays at times that may
  not be convenient in our release schedule.

On the plus side, Travis-CI's integration with github -- automatically
testing all pending pull requests -- is exceptional.

**Builds**

There is no centralized effort for automated binary builds for
matplotlib.  However, the following disparate things are being done
[If the authors mentioned here could fill in detail, that would be
great!]:

- @sandrotosi: builds Debian packages

- @takluyver: Has automated Ubuntu builds on Launchpad

- @cgohlke: Makes Windows builds (don't know how automated that is)

- @r-owen: Makes OS-X builds (don't know how automated that is)

**Documentation**

Documentation of master is now built by travis and uploaded to http://matplotlib.org/devdocs/index.html

@NelleV, I believe, generates the docs automatically and posts them on
the web to chart MEP10 progress.

Peculiarities of matplotlib
---------------------------

matplotlib has complex requirements that make testing and building
more taxing than many other Python projects.

- The CPU time to run the tests is quite high.  It puts us beyond the
  free accounts of many CI services (e.g. ShiningPanda)

- It has a large number of dependencies, and testing the full matrix
  of all combinations is impractical.  We need to be clever about what
  space we test and guarantee to support.

Requirements
------------

This section outlines the requirements that we would like to have.

#. Testing all pull requests by hooking into the Github API, as
   Travis-CI does

#. Testing on all major platforms: Linux, Mac OS-X, MS Windows (in
   that order of priority, based on user survey)

#. Retain the last n days worth of build and test products, to aid in
   post-mortem debugging.

#. Automated nightly binary builds, so that users can test the
   bleeding edge without installing a complete compilation
   environment.

#. Automated benchmarking.  It would be nice to have a standard
   benchmark suite (separate from the tests) whose performance could
   be tracked over time, in different backends and platforms.  While
   this is separate from building and testing, ideally it would run on
   the same infrastructure.

#. Automated nightly building and publishing of documentation (or as
   part of testing, to ensure PRs don't introduce documentation bugs).
   (This would not replace the static documentation for stable
   releases as a default).

#. The test systems should be manageable by multiple developers, so
   that no single person becomes a bottleneck.  (Travis-CI's design
   does this well -- storing build configuration in the git
   repository, rather than elsewhere, is a very good design.)

#. Make it easy to test a large but sparse matrix of different
   versions of matplotlib's dependencies.  The matplotlib user survey
   provides some good data as to where to focus our efforts:
   https://docs.google.com/spreadsheet/ccc?key=0AjrPjlTMRTwTdHpQS25pcTZIRWdqX0pNckNSU01sMHc#gid=0

#. Nice to have: A decentralized design so that those with more
   obscure platforms can publish build results to a central dashboard.

Implementation
==============

This part is yet-to-be-written.

However, ideally, the implementation would be a third-party service,
to avoid adding system administration to our already stretched time.
As we have some donated funds, this service may be a paid one if it
offers significant time-saving advantages over free offerings.

Backward compatibility
======================

Backward compatibility is not a major concern for this MEP.  We will
replace current tools and procedures with something better and throw
out the old.

Alternatives
============


Hangout Notes
=============

CI Infrastructure
-----------------

- We like Travis and it will probably remain part of our arsenal in
  any event.  The reliability issues are being looked into.

- Enable Amazon S3 uploads of testing products on Travis.  This will
  help with post-mortem of failures (@mdboom is looking into this
  now).

- We want Mac coverage.  The best bet is probably to push Travis to
  enable it for our project by paying them for a Pro account (since
  they don't otherwise allow testing on both Linux and Mac).

- We want Windows coverage.  Shining Panda is an option there.

- Investigate finding or building a tool that would collect and
  synthesize test results from a number of sources and post it to
  Github using the Github API.  This may be of general use to the
  Scipy community.

- For both Windows and Mac, we should document (or better yet, script)
  the process of setting up the machine for a build, and how to build
  binaries and installers.  This may require getting information from
  Russel Owen and Christoph Gohlke.  This is a necessary step for
  doing automated builds, but would also be valuable for a number of
  other reasons.

The test framework itself
-------------------------

- We should investigate ways to make it take less time

   - Eliminating redundant tests, if possible

   - General performance improvements to matplotlib will help

- We should be covering more things, particularly more backends

- We should have more unit tests, fewer integration tests, if possible
