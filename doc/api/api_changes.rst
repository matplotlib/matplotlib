
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
added to Matplotlib, see :ref:`whats-new`

.. ifconfig:: releaselevel == 'dev'

   .. note::

      The list below is a table of contents of individual files from the
      most recent :file:`api_changes_X.Y` folder.

      When a release is made

       - The include directive below should be changed to point to the new file
         created in the previous step.


   .. toctree::
      :glob:
      :maxdepth: 1

      next_api_changes/behavior/*
      next_api_changes/deprecations/*
      next_api_changes/development/*
      next_api_changes/removals/*

.. include:: prev_api_changes/api_changes_3.4.2.rst
.. include:: prev_api_changes/api_changes_3.4.0.rst
