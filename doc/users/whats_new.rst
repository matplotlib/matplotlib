.. _whats-new:

===============================
 What's new in Matplotlib 3.0.3
===============================

For a list of all of the issues and pull requests since the last
revision, see the :ref:`github-stats`.

.. contents:: Table of Contents
  :depth: 4

..
   For a release, add a new section after this, then comment out the include
   and toctree below by indenting them. Uncomment them after the release.

   .. include:: next_whats_new/README.rst

   .. toctree::
      :glob:
      :maxdepth: 1

      next_whats_new/*


axes_grid1 and axisartist Axes no longer draw spines twice
``````````````````````````````````````````````````````````

Previously, spines of `axes_grid1` and `axisartist` Axes would be drawn twice,
leading to a "bold" appearance.  This is no longer the case.


==================
Previous Whats New
==================

.. toctree::
   :glob:
   :maxdepth: 1
   :reversed:

   prev_whats_new/changelog
   prev_whats_new/whats_new_*
