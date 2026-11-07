Data Containers Architecture
============================

Overview
--------

Historically, Matplotlib :class:`Artist` s have been implemented with bespoke
Numpy arrays (or array-like objects, somewhat more loosely).
This makes it so that there is often very little internal consistency between
Artist types, with various schemes of caching, updating, or accessing the
arrays that Artists hold as their source of truth.

Data Containers are a system to allow Matplotlib artists to have a more
consistent and extensible data interface.
The key idea is that there is a new :class:`DataContainer` datatype that
:class:`Artist` s can use which provides a consistent interface and allows for such
things as caching, draw-time updating, key mapping, etc.

This project fundamentally comes in two parts, the Data Containers themselves,
and an execution graph which transforms input data through a pipeline of
operations, ultimately resulting in the representation of the data on a
display.

In the initial implementation phase, the execution graph is considered
entirely internal to Matplotlib, and users are not expected to interact with it
directly.

Data Containers
---------------

The Data Containers themselves are a Protocol which describes just two
methods, :meth:`DataContainer.query` and :meth:`DataContainer.describe`.
Any Python object which provides these two methods (and appropriately returns
the expected data) can be used as a Data Container.

In practice, naive wrappers for common data sources (such as numpy arrays,
dataframes, etc) are easy to write.
Matplotlib Artists have specialized containers associated with them that handle
updating their data to support backwards compatibility.

Query
^^^^^

The primary method of Data Containers is :meth:`DataContainer.query`, which is what provides the data to a caller.
This method takes two parameters, a :class:`Graph` and a a coordinate of the parent (typically ``"axes"`` or ``"figure"``).
Many DataContainers, particularly those that represent static data, do not need either of these parameters.
The parameters are primarily there for :class:`DataContainer` objects that represent data that is dynamically computed in relation to the viewport of their axes, such as :class:`FuncContainer`.

The :meth:`DataContainer.query` returns both a dictionary mapping string keys to arbitrary (though most commonly numeric array) data, and a cache key.

.. code-block:: python

   >>> th = np.linspace(0, 2 * np.pi, 16)
   >>> ac = ArrayContainer(x=th, y=np.cos(th))
   >>> ac.describe()
   {'x': Desc(shape=(16,), coordinates='auto'), 'y': Desc(shape=(16,), coordinates='auto')}
   >>> # For an ArrayContainer, the Graph and parent coordinates are ignored,
   >>> # So pass an empty Graph and "axes" just to demonstrate a typical call
   >>> ac.query(Graph([]), "axes")
   (
       {
           'x': array([0.        , 0.41887902, 0.83775804, 1.25663706, 1.67551608,
          2.0943951 , 2.51327412, 2.93215314, 3.35103216, 3.76991118,
          4.1887902 , 4.60766923, 5.02654825, 5.44542727, 5.86430629,
          6.28318531]),
           'y': array([ 1.        ,  0.91354546,  0.66913061,  0.30901699, -0.10452846,
          -0.5       , -0.80901699, -0.9781476 , -0.9781476 , -0.80901699,
          -0.5       , -0.10452846,  0.30901699,  0.66913061,  0.91354546,
           1.        ])
       },
       '62783713-4725-477a-96cb-0fe7e6abef74'
   )


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
A variable should, in all instances, represent a single integer within a
container.
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


Graph
^^^^^

The graph consists of a set of :class:`Edge` objects.

The primary method of a graph, :meth:`Graph.evaluator`, which implements a version of Djikstra's algorithm to return a sequence of edges that take a given input (usually the :meth:`DataContainer.describe` of a given data container) and achieve the desired output, which is similarly a dictionary of :class:`Desc` objects with str keys, specifying the desired output keys and coordinate space.

A typical call may look like:

.. code-block:: python

   >>> graph.evaluator(container.describe(), {"x": Desc("(N,)", "display"), "y": Desc("(N,)", "display")})
   SequenceEdge(...)

Internally, the :class:`Graph` is able to have separate subgraphs for keys that are not interconnected (e.g. often many individual keys have a linear subgraph that does not depend on other data from the data container). This is done for efficiency, but does not actually affect the result.


Edges
^^^^^

An :class:`Edge`, generically, takes a set of inputs, which are a dictionary of string keys to :class:`Desc` objects, and produces a set of outputs (a similar dictionary).
Edges are in theory incredibly flixible and generalize the transformation from
user data to rendering beyond the spatial transformations, size scaling in
:class:`SizedCollection` and and color mapping via :class:`ColorMappable`.
For example, this will enable mapping of user data to marker shape and allow
every artist that supports color to be colormappable.
However, in the initial implementation, the edges represent a minimal set of the Matplotlib spatial transform stack.
These are the transforms which allow for coordinate changes between "data", "axes", and "figure" space, which will be expanded in the future.
