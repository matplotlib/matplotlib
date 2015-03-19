
MEP25: Serialization
====================


Status
------

**Discussion**

Branches and Pull requests
--------------------------

* development branches:

* related pull requests:

Abstract
--------

This MEP aims at adding a serializable ``Controller`` objects to act as an ``Artist`` managers. Users would then communicate changes to an ``Artist`` via a ``Controller``. In this way, functionality of the ``Controller`` objects may be added incrementally since each ``Artist`` is still responsible for drawing everything. The goal is to create an API that is usable both by graphing libraries requiring high-level descriptions of figures and libraries requiring low-level interpretations.

Detailed description
--------------------

Matplotlib is a core plotting engine with an API that many users already understand. It's difficult/impossible for other graphing libraries to (1) get a complete figure description, (2) output raw data from the figure object as the user has provided it, (3) understand the semantics of the figure objects without heuristics, and (4) give matplotlib a complete figure description to visualize. In addition, because an ``Artist`` has no conception of its own semantics within the figure, it's difficult to interact with them in a natural way.

In this sense, matplotlib will adopt a standard Model-View-Controller (MVC) framework. The *Model* will be the user defined data, style, and semantics. The *Views* are the ensemble of each individual ``Artist``, which are responsible for producing the final image based on the *model*. The *Controller* will be the ``Controller`` object managing its set of ``Artist`` objects.

The ``Controller`` must be able to export the information that it's carrying about the figure on command, perhaps via a ``to_json`` method or similar. Because it would be extremely extraneous to duplicate all of the information in the model with the controller, only user-specified information (data + style) are explicitly kept. If a user wants more information (defaults) from the view/model, it should be able to query for it.

- This might be annoying to do, non-specified kwargs are pulled from the rcParams object which is in turn created from reading a user specified file and can be dynamically changed at run time.  I suppose we could keep a dict of default defaults and compare against that. Not clear how this will interact with the style sheet [[MEP26]] - @tacaswell

Additional Notes:

* The `raw data` does not necessarily need to be a ``list``, ``ndarray``, etc. Rather, it can more abstractly just have a method to yield data when needed.

* Because the ``Controller`` will contain extra information that users may not want to keep around, it should *not* be created by default. You should be able to both (a) instantiate a ``Controller`` with a figure and (b) build a figure with a ``Controller``.

Use Cases:

* Export all necessary informat
