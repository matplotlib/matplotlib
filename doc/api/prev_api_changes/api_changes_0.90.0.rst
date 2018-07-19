Changes for 0.90.0
==================

.. code-block:: text

    All artists now implement a "pick" method which users should not
    call.  Rather, set the "picker" property of any artist you want to
    pick on (the epsilon distance in points for a hit test) and
    register with the "pick_event" callback.  See
    examples/pick_event_demo.py for details

    Bar, barh, and hist have "log" binary kwarg: log=True
    sets the ordinate to a log scale.

    Boxplot can handle a list of vectors instead of just
    an array, so vectors can have different lengths.

    Plot can handle 2-D x and/or y; it plots the columns.

    Added linewidth kwarg to bar and barh.

    Made the default Artist._transform None (rather than invoking
    identity_transform for each artist only to have it overridden
    later).  Use artist.get_transform() rather than artist._transform,
    even in derived classes, so that the default transform will be
    created lazily as needed

    New LogNorm subclass of Normalize added to colors.py.
    All Normalize subclasses have new inverse() method, and
    the __call__() method has a new clip kwarg.

    Changed class names in colors.py to match convention:
    normalize -> Normalize, no_norm -> NoNorm.  Old names
    are still available for now.

    Removed obsolete pcolor_classic command and method.

    Removed lineprops and markerprops from the Annotation code and
    replaced them with an arrow configurable with kwarg arrowprops.
    See examples/annotation_demo.py - JDH
