
=============
 API Changes
=============

A log of changes to the most recent version of Matplotlib that affect the
outward-facing API. If updating Matplotlib breaks your scripts, this list may
help you figure out what caused the breakage and how to fix it by updating
your code. For API changes in older versions see :doc:`api_changes_old`.

For new features that were added to Matplotlib, see :ref:`whats-new`.

This pages lists API changes for the most recent version of Matplotlib.

.. toctree::
   :maxdepth: 1

   api_changes_old

..

   .. note::

     The list below is a table of contents of individual files from the 'next_api_changes' folder.
     When a release is made

       - The full text list below should be moved into its own file in 'prev_api_changes'
       - All the files in 'next_api_changes' should be moved to the bottom of this page
       - This note, and the toctree below should be commented out


      .. toctree::
         :glob:
         :maxdepth: 1

         next_api_changes/*

API Changes for 3.0.3
=====================

matplotlib.font_manager.win32InstalledFonts return value
--------------------------------------------------------

`matplotlib.font_manager.win32InstalledFonts` returns an empty list instead
of None if no fonts are found.


Matplotlib.use now has an ImportError for interactive backend
-------------------------------------------------------------

Switching backends via `matplotlib.use` is now allowed by default,
regardless of whether `matplotlib.pyplot` has been imported. If the user
tries to switch from an already-started interactive backend to a different
interactive backend, an ImportError will be raised.
