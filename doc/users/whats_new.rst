.. _whats-new:

===========
What's new?
===========

.. ifconfig:: releaselevel == 'dev'

   .. note::

      The list below is a table of contents of individual files from the
      'next_whats_new' folder.

      When a release is made

       - All the files in 'next_whats_new/' should be moved to a single file in
         'prev_whats_new/'.
       - The include directive below should be changed to point to the new file
         created in the previous step.


   .. toctree::
      :glob:
      :maxdepth: 1

      next_whats_new/*

.. Be sure to update the version in `exclude_patterns` in conf.py.
.. include:: prev_whats_new/whats_new_3.4.0.rst
