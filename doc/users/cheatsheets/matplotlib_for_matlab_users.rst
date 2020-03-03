.. _matplotlib-for-matlab-users:

================================
WIP: Matplotlib for Matlab users
================================

Introduction
============

This page is intended to be a reference for new matplotlib users who are coming
from MATLAB®.

.. raw:: html

   <style>
   table.docutils td { border: solid 1px #ccc; }
   </style>

First Things First
====================
WIP:

Insert two code blocks: one being the "hello world" plot from matplotlib and
the other one from MATLAB®. It'll be nice to show code and figures side
by side in a two-column table. But I'm not sure how easy it is to acheive it
using RST.


Table of Rough MATLAB-Matplotlib Equivalents
============================================
.. list-table::
   :header-rows: 1

   * - MATLAB
     - NumPy
     - Notes

   * - ``ylim([-2 2])``
     - ``set_ylim(-2, 2)``
     - Set y-limit

   * - ``figure('position', [0, 0, 500, 200])``
     - ``plt.figure(figsize=(6,3))``
     - Create a figure with specific size. MATLAB: left bottom width height;
       Matplotlib: width height


Some Key Differences
====================
.. list-table::
   :header-rows: 1

   * - MATLAB
     - NumPy
   * - placeholder
     - placeholder

Credits
================
This guide is inpired by its counterpart in Scipy/Numpy:
https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html, which is
another an excellent read for users coming from MATLAB®.

MATLAB® is registered trademarks of The MathWorks.
