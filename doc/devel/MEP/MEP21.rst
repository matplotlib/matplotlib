==============================
 MEP21: color and cm refactor
==============================

.. contents::
   :local:


Status
======

- **Discussion**: This MEP has not commenced yet, but here are some
  ongoing ideas which may become a part of this MEP:



Branches and Pull requests
==========================



Abstract
========


* color

  * tidy up the namespace
  * Define a "Color" class
  * make it easy to convert from one color type to another ```hex ->
    RGB```, ```RGB -> hex```, ```HSV -> RGB``` etc.
  * improve the construction of a colormap - the dictionary approach
    is archaic and overly complex (though incredibly powerful)
  * make it possible to interpolate between two or more color types
    in different modes, especially useful for construction of
    colormaps in HSV space for instance

* cm

  * rename the module to something more descriptive - mappables?


Overall, there are a lot of improvements that can be made with
matplotlib color handling - managing backwards compatibility will be
difficult as there are some badly named variables/modules which really
shouldn't exist - but a clear path and message for migration should be
available, with a large amount of focus on this in the API changes
documentation.


Detailed description
====================

Implementation
==============


Backward compatibility
======================

Alternatives
============
