:orphan:

.. NOTE TO EDITORS OF THIS FILE
   This file serves as the README directly available in the file system next to the
   next_whats_new entries. The content between the ``whats-new-guide-*`` markers is
   additionally included in the documentation page ``doc/devel/api_changes.rst``. Please
   check that the page builds correctly after changing this file.


Instructions for writing "What's new" entries
=============================================

.. whats-new-guide-start

Please place new portions of :file:`whats_new.rst` in the
:file:`doc/users/next_whats_new/` directory.

When adding an entry please look at the currently existing files to
see if you can extend any of them.  If you create a file, name it
something like :file:`cool_new_feature.rst` if you have added a brand new
feature or something like :file:`updated_feature.rst` for extensions of
existing features.

Include contents of the form::

    Section title for feature
    -------------------------

    A bunch of text about how awesome the new feature is and examples of how
    to use it.

    A sub-section
    ~~~~~~~~~~~~~

Please avoid using references in section titles, as it causes links to be
confusing in the table of contents. Instead, ensure that a reference is
included in the descriptive text.

.. whats-new-guide-end
