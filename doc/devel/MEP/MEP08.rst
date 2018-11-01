============
 MEP8: PEP8
============

.. contents::
   :local:


Status
======

**Completed**

We are currently enforcing a sub-set of pep8 on new code contributions.

Branches and Pull requests
==========================

None so far.

Abstract
========

The matplotlib codebase predates PEP8, and therefore is less than
consistent style-wise in some areas.  Bringing the codebase into
compliance with PEP8 would go a long way to improving its legibility.

Detailed description
====================

Some files use four space indentation, some use three.  Some use
different levels in the same file.

For the most part, class/function/variable naming follows PEP8, but it
wouldn't hurt to fix where necessary.

Implementation
==============

The implementation should be fairly mechanical: running the pep8 tool
over the code and fixing where appropriate.

This should be merged in after the 2.0 release, since the changes will
likely make merging any pending pull requests more difficult.

Additionally, and optionally, PEP8 compliance could be tracked by an
automated build system.

Backward compatibility
======================

Public names of classes and functions that require change (there
shouldn't be many of these) should first be deprecated and then
removed in the next release cycle.

Alternatives
============

PEP8 is a popular standard for Python code style, blessed by the
Python core developers, making any alternatives less desirable.
