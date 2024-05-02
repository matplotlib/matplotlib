BackendRegistry
~~~~~~~~~~~~~~~

New :class:`~matplotlib.backends.registry.BackendRegistry` class is the single
source of truth for available backends. The singleton instance is
``matplotlib.backends.backend_registry``. It is used internally by Matplotlib,
and also IPython (and therefore Jupyter) starting with IPython 8.24.0.

There are three sources of backends: built-in (source code is within the
Matplotlib repository), explicit ``module://some.backend`` syntax (backend is
obtained by loading the module), or via an entry point (self-registering
backend in an external package).

To obtain a list of all registered backends use:

    >>> from matplotlib.backends import backend_registry
    >>> backend_registry.list_all()
