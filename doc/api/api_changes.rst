
===========
API Changes
===========

If updating Matplotlib breaks your scripts, this list may help you figure out
what caused the breakage and how to fix it by updating your code.

For API changes in older versions see

.. toctree::
   :maxdepth: 1

   api_changes_old

Changes for the latest version are listed below. For new features that were
added to Matplotlib, see :ref:`whats-new`.

.. ifconfig:: releaselevel == 'dev'

   .. note::

      The list below is a table of contents of individual files from the
      'recent_api_changes_3.3' folder.

      When a release is made

       - The files in 'recent_api_changes_3.3/' should be moved to a new file in
         'prev_api_changes/'.
       - The include directive below should be changed to point to the new file
         created in the previous step.


   .. toctree::
      :glob:
      :maxdepth: 1

      recent_api_changes_3.3/*

.. include:: prev_api_changes/api_changes_3.2.0.rst
