:orphan:

.. NOTE TO EDITORS OF THIS FILE
   This file serves as the README directly available in the file system next to the
   next_whats_new entries. The content between the ``whats-new-guide-*`` markers is
   additionally included in the documentation page ``doc/devel/api_changes.rst``. Please
   check that the page builds correctly after changing this file.


Instructions for writing "What's new" entries
=============================================
.. whats-new-guide-start

Each new feature (e.g. function, parameter, rcParam, config value, behavior,
...) must be described through a "What's new" entry.

Each entry is written into a separate file in the
:file:`doc/release/next_whats_new/` directory. They are sorted and merged into
:file:`whats_new.rst` during the release process.

When adding an entry please look at the currently existing files to
see if you can extend any of them.  If you create a file, name it
something like :file:`cool_new_feature.rst` if you have added a brand new
feature or something like :file:`updated_feature.rst` for extensions of
existing features.

Include contents of the form::

    Section title for feature
    -------------------------

    A description of the feature from the user perspective. This should include
    what the feature allows users to do and how the feature is used. Technical
    details should be left out when they do not impact usage, for example
    implementation details.

    The description may include a short instructive example, if it helps to
    understand the feature.

Please avoid using references in section titles, as it causes links to be
confusing in the table of contents. Instead, ensure that a reference is
included in the descriptive text. Use inline literals (double backticks)
to denote code objects in the title.



.. whats-new-guide-end
