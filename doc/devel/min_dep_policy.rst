.. _min_deps_policy:

======================================
Minimum Version of Dependencies Policy
======================================

For the purpose of this document, 'minor version' is in the sense of
SemVer (major, minor, patch) and includes both major and minor
releases. For projects that use date-based versioning, every release
is a 'minor version'.


Python and numpy
================

- support minor versions of ``Python`` initially released
  36 months prior to our planned release date.
- support minor versions of ``numpy`` initially released in the 36
  months prior to our planned release date or oldest that supports the
  minimum python version (which ever is higher)

We will bump the minimum python and numpy versions as we can every
minor and major release, but never on a patch release.

Python Dependencies
===================

For python dependencies we should support at least:

with compiled extensions
  minor versions initially released in the 36 months prior to our
  planned release date or the oldest that support our minimum python +
  numpy

without complied extensions
  minor versions initially released in the 24 months prior to our
  planed release date or the oldest that supports our minimum python.

We will only bump these dependencies as we need new features or the
old versions no longer support our minimum numpy or python.


System and C-dependencies
=========================

For system or c-dependencies (libpng, freetype, GUI frameworks, latex,
gs, ffmpeg) support as old as practical.  These can be difficult to
install for end-users and we want to be usable on as many systems as
possible.  We will bump these on a case-by-case basis.
