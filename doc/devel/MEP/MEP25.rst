
MEP25: Serialization
====================
.. contents::
   :local:

Status
------

**Discussion**

Branches and Pull requests
--------------------------

* development branches:

* related pull requests:

Abstract
--------

This MEP aims at adding a serializable ``Controller`` objects to act
as an ``Artist`` managers. Users would then communicate changes to an
``Artist`` via a ``Controller``. In this way, functionality of the
``Controller`` objects may be added incrementally since each
``Artist`` is still responsible for drawing everything. The goal is to
create an API that is usable both by graphing libraries requiring
high-level descriptions of figures and libraries requiring low-level
interpretations.

Detailed description
--------------------

Matplotlib is a core plotting engine with an API that many users
already understand. It's difficult/impossible for other graphing
libraries to (1) get a complete figure description, (2) output raw
data from the figure object as the user has provided it, (3)
understand the semantics of the figure objects without heuristics,
and (4) give matplotlib a complete figure description to visualize. In
addition, because an ``Artist`` has no conception of its own semantics
within the figure, it's difficult to interact with them in a natural
way.

In this sense, matplotlib will adopt a standard
Model-View-Controller (MVC) framework. The *Model* will be the user
defined data, style, and semantics. The *Views* are the ensemble of
each individual ``Artist``, which are responsible for producing the
final image based on the *model*. The *Controller* will be the
``Controller`` object managing its set of ``Artist`` objects.

The ``Controller`` must be able to export the information that it's
carrying about the figure on command, perhaps via a ``to_json`` method
or similar. Because it would be extremely extraneous to duplicate all
of the information in the model with the controller, only
user-specified information (data + style) are explicitly kept. If a
user wants more information (defaults) from the view/model, it should
be able to query for it.

- This might be annoying to do, non-specified kwargs are pulled from
  the rcParams object which is in turn created from reading a user
  specified file and can be dynamically changed at run time.  I
  suppose we could keep a dict of default defaults and compare against
  that. Not clear how this will interact with the style sheet
  [[MEP26]] - @tacaswell

Additional Notes:

* The `raw data` does not necessarily need to be a ``list``,
  ``ndarray``, etc. Rather, it can more abstractly just have a method
  to yield data when needed.

* Because the ``Controller`` will contain extra information that users
  may not want to keep around, it should *not* be created by
  default. You should be able to both (a) instantiate a ``Controller``
  with a figure and (b) build a figure with a ``Controller``.

Use Cases:

* Export all necessary informat
* Serializing a matplotlib figure, saving it, and being able to rerun later.
* Any other source sending an appropriately formatted representation to matplotlib to open

Examples
--------
Here are some examples of what the controllers should be able to do.

1. Instantiate a matplotlib figure from a serialized representation (e.g., JSON): ::

    import json
    from matplotlib.controllers import Controller
    with open('my_figure') as f:
        o = json.load(f)
    c = Controller(o)
    fig = c.figure

2. Manage artists from the controller (e.g., Line2D): ::

    # not really sure how this should look
    c.axes[0].lines[0].color = 'b'
    # ?

3. Export serializable figure representation: ::

    o = c.to_json()
    # or... we should be able to throw a figure object in there too
    o = Controller.to_json(mpl_fig)

Implementation
--------------

1. Create base ``Controller`` objects that are able to manage
   ``Artist`` objects (e.g., ``Hist``)

    Comments:

    * initialization should happen via unpacking ``**``, so we need a
      copy of call signature parameter for the ``Artist`` we're
      ultimately trying to control. Unfortunate hard-coded
      repetition...
    * should the additional ``**kwargs`` accepted by each ``Artist``
      be tracked at the ``Controller``
    * how does a ``Controller`` know which artist belongs where? E.g.,
      do we need to pass ``axes`` references?

    Progress:

    * A simple NB demonstrating some functionality for
      ``Line2DController`` objects:
      http://nbviewer.ipython.org/gist/theengineear/f0aa8d79f64325e767c0

2. Write in protocols for the ``Controller`` to *update* the model.

    Comments:

    * how should containers be dealt with? E.g., what happens to old
      patches when we re-bin a histogram?
    * in the link from (1), the old line is completely destroyed and
      redrawn, what if something is referencing it?

3. Create method by which a json object can be assembled from the
   ``Controllers``
4. Deal with serializing the unserializable aspects of a figure (e.g.,
   non-affine transforms?)
5. Be able to instantiate from a serialized representation
6. Reimplement the existing pyplot and Axes method,
   e.g. ``pyplot.hist`` and ``Axes.hist`` in terms of the new
   controller class.

> @theengineer: in #2 above, what do you mean by *get updates* from
each ``Artist``?

^ Yup. The ``Controller`` *shouldn't* need to get updated. This just
happens in #3. Delete comments when you see this.

Backward compatibility
----------------------

* pickling will change
* non-affine transformations will require a defined pickling method

Alternatives
------------

PR #3150 suggested adding semantics by parasitically attaching extra
containers to axes objects. This is a more complete solution with what
should be a more developed/flexible/powerful framework.
