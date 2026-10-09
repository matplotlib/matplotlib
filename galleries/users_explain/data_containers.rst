Data Containers Architecture
============================

Overview
--------

Data Containers are a system to allow Matplotlib artists to have a more
consistent data interface.
The key idea is that there is a new :class:`DataContainer` datatype that
Artists can use which provides a consistent interface and allows for such
things as caching, draw-time updating, key mapping, etc.

This project fundamentally comes in two parts, the Data Containers themselves,
and an execution graph which transforms input data through a pipeline of
operations, ultimately resulting in the representation of the data on a
display.

In the initial phase of rolling these out, the execution graph is considered
entirely internal to Matplotlib, and users are not expected to interact with it
directly.

Data Containers
---------------

The Data Containers themselves are actually a Protocol which describes just two
methods, :meth:`DataContainer.query` and :meth:`DataContainer.describe`.
Any Python object which provides these two methods (and appropriately returns
the expected data) can be used as a Data Container.

In practice, small wrappers for common data sources (such as numpy arrays,
dataframes, etc) are easy to write.
Matplotlib Artists have specialized containers associated with them that handle
backwards compatibility and other common operations for data related to that
artist.

Query
^^^^^

The primary method of Data Containers is :meth:`DataContainer.query`, which is what provides the data to a caller.
This method takes two parameters, a :class:`Graph` and a a coordinate of the parent (typically ``"axes"`` or ``"figure"``).
Many DataContainers, particularly those that represent static data, do not need either of these parameters.
The parameters are primarily there for :class:`DataContainer` objects that represent data that is dynamically computed in relation to the viewport of their axes, such as :class:`FuncContainer`.

The :meth:`DataContainer.query` returns both a dictionary mapping string keys to arbitrary (though most commonly numeric array) data, and a cache key.


Describe
^^^^^^^^

Since :meth:`DataContainer.query` can do relatively long computation (or otherwise high latency operations), a :class:`DataContainer` also has a method to :meth:`DataContainer.describe` the data.
This returns a dictionary mapping string keys to :class:`Desc` objects.

The purpose of this is to be able to quickly validate what operations are available, given a set of data.
This accounts for factors such as shape consistency and coordinate systems.

Desc objects
^^^^^^^^^^^^

:class:`Desc` objects are a dataclass that contains a ``shape`` and a ``coordinates``.

``shape`` is a tuple of integer or strings.
In its simplest form, this is just the shape of an array (or an empty tuple for scalar data).
However, `Desc` objects also allow variables to be used in shape descriptions.
This allows, for instance, data containers to report that a given field is an ``("M", "N")`` array.
Variables must be single characters.
It is expected that all variables from a given data container will resolve to a single number once queried.
That is, two fields that report ``("N",)`` will be the same length and ``("M", "M")`` will be a square array.
Additionally, additive offsets can be used, such as ``("N", "N+1")``, which represents an array that is one larger in the second dimension (e.g. a ``(4, 5)`` array).

Coordinates allow for differentiation of semantic meaning of a given variable.
The classic example is the difference between "data", "axes", "figure", and "display" coordinates.
However, this concept is extensible, including to non-spatial coordinates such as color spaces.


Execution Graph
---------------

The second portion of the Data Containers project is an execution graph.
In its initial implementation, it is intended for internal use, and thus not intended to be directly interacted with by end users.
That said, the core functionality of the execution graph is implemented and can be used by controlled, internal to Matplotlib, artists.


Edges
^^^^^

An :class:`Edge`, generically, takes a set of inputs, which are a dictionary of string keys to :class:`Desc` objects, and produces a set of outputs (a similar dictionary).
In theory, Edges are incredibly flexible, and can do many operations, including things such as resizing, upscaling/downscaling arrays, applying color transformations, etc.
In practice, for the initial implementation, the edges represent a relatively minimal set of the Matplotlib transform stack.
These are the transforms which allow for coordinate changes between "data", "axes", and "figure" space, for instance.


Graph
^^^^^

The graph consists of a set of :class:`Edge` objects.

The primary method of a graph, :meth:`Graph.evaluator`, which implements a version of Djikstra's algorithm to return a sequence of edges that take a given input (usually the :meth:`DataContainer.describe` of a given data container) and achieve the desired output, which is similarly a dictionary of :class:`Desc` objects with str keys.

Internally, the :class:`Graph` is able to have separate subgraphs for keys that are not interconnected (e.g. often many individual keys have a linear subgraph that does not depend on other data from the data container). This is done for efficiency, but does not actually affect the result.
