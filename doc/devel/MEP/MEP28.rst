=======================
 MEP 28 Artist Refactor
=======================

.. contents::
   :local:


Branches and Pull requests
==========================

All development branches containing work on this MEP should be linked to from here.

All pull requests submitted relating to this MEP should be linked to
from here.  (A MEP does not need to be implemented in a single pull
request if it makes sense to implement it in discrete phases).

Abstract
========

During the course of MPL, the number and complexity of artists have grown, to become quite unmanageable, this PR seeks to address that by refactoring the Artists and creating a more sensible class structure.


Detailed description
====================

The goal of this MEP lies in automating and reducing as much of the "admin" code as possible making it easier to see what goes on and where and make faster changes to the codebase.  A lot of the artist code for the example gets taken up with simple getter/setters, very laborious, and when it comes to the documentation it looks like a getter/setter desert. More so, we the same properties duplicated throughout the codebase, such as fillcolor, where we have ``set_fillcolor`` methods in ``lines``, ``markers`` and ``patches``.  This makes readability and maintainability very difficult and just adds to the Artist doc desert.

With a clearer structure it also makes it easier to add new features such as keeping the legend upto date with its parent artists, add make it easier to implement dynamical property changing.  


Implementation
==============

## Step 1 - Remove getter/setters
The easiest and biggest reduction comes from removing all of the ``artist.set_xxx(value)`` methods (where possible).
We will maintain a list of the properties the Artist class held as a class property.
To maintain BC we will add a single ``__getattr__`` in the Artist base class to return a closure to set a dict property.

With this done, we can easily create property getter/setters, i.e. ``artist.xxx = value`` with  ``__setattr__``.
Create the getterHere we plan to use dictionary to manage the Artist properties.

The benefit of this, we can mark Artists as "stale" in a single place.


## Step 2 - Class inheritance
With a reduced codebase it becomes easier to assess how to split the classes.

To start with we shall use the following refactor rule:
If a keyword gets used by multiple artists that don't have direct ancestry, then it needs to go in a separate class.  This prevents super keyword conflicts, but more importantly leads to good OOP design as it encourages full encapsulation of classes and a clearer API.

Once we have this cleaner structure we can then reassess this MEP to see what problems remain.


## Step 3 - Legend refactoring
As a natural consequence of the above steps, this will lead to a simplification of the legend.  Using sets we can fine-tune which properties get updated.


Backward compatibility
======================

If we address the issue of alpha stacking in this PR then that would break BC, but we shall cross that bridge when we get to it.
