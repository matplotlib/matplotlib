.. _min_deps_policy:

======================================
Minimum Version of Dependencies Policy
======================================

Python and numpy
================

- support minor versions of ``Python`` initially released in the previous
  3 years
- support minor versions of ``numpy`` initially released in the
  previous 3 years or oldest that supports the minimum python version
  (which ever is higher)

We will bump the minimum python and numpy versions as we can every
minor and major release, but never on a patch release.

Python Dependencies
===================

For python dependencies we should support at least

with compiled extensions
  minor versions released in the last 3 years
  or the oldest that support our minimum python + numpy

without complied extensions
  minor versions released in the last 2 years or the oldest that
  supports our minimum python.

We will only bump these dependencies as we need new features or the
old versions no longer support our minimum numpy or python.


System and C-dependencies
=========================

For system or c-dependencies (libpng, freetype, GUI frameworks, latex,
gs, ffmpeg) support as old as practical.  These can be difficult to
install for end-users and we want to be usable on as many systems as
possible.  We will bump these on a case-by-case basis.
