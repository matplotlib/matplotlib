:orphan:

Adding API change notes
=======================

Please place new portions of `api_changes.rst` in the
`next_api_changes` directory.

When adding an entry please look at the currently existing files to
see if you can extend any of them.  If you create a file, name it
:file:`what_api_changes.rst` (ex :file:`deprecated_rcparams.rst`) with
contents following the form: ::

    Brief description of change
    ---------------------------

    Long description of change, justification, and work-arounds to
    maintain old behavior (if any).


If you need more heading levels, please use ``~~~~`` and ``++++``.
