:orphan:

.. NOTE TO EDITORS OF THIS FILE
   This file serves as the README directly available in the file system next to the
   next_api_changes entries. The content between the ``api-change-guide-*`` markers is
   additionally included in the documentation page ``doc/devel/api_changes.rst``. Please
   check that the page builds correctly after changing this file.

Adding API change notes
=======================

.. api-change-guide-start

API change notes for future releases are collected in :file:`doc/api/next_api_changes/`.
They are divided into four subdirectories:

- **Deprecations**: Announcements of future changes. Typically, these will
  raise a deprecation warning and users of this API should change their code
  to stay compatible with future releases of Matplotlib. If possible, state
  what should be used instead.
- **Removals**: Parts of the API that got removed. If possible, state what
  should be used instead.
- **Behaviour changes**: API that stays valid but will yield a different
  result.
- **Development changes**: Changes to the build process, dependencies, etc.

Please place new entries in these directories with a new file named
``99999-ABC.rst``, where ``99999`` would be the PR number, and ``ABC`` the
author's initials. Typically, each change will get its own file, but you may
also amend existing files when suitable. The overall goal is a comprehensible
documentation of the changes.

A typical entry could look like this::

   Locators
   ~~~~~~~~
   The unused `Locator.autoscale()` method is deprecated (pass the axis
   limits to `Locator.view_limits()` instead).

Please avoid using references in section titles, as it causes links to be
confusing in the table of contents. Instead, ensure that a reference is
included in the descriptive text.

.. api-change-guide-end
