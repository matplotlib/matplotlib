.. redirect-from:: /users/explain/writing_a_backend_pyplot_interface

.. _writing_backend_interface:

=========================================
Writing a backend -- the pyplot interface
=========================================

This page assumes general understanding of the information in the
:ref:`backends` page, and is instead intended as reference for
third-party backend implementers.  It also only deals with the interaction
between backends and `.pyplot`, not with the rendering side, which is described
in `.backend_template`.

There are two APIs for defining backends: a new canvas-based API (introduced in
Matplotlib 3.6), and an older function-based API.  The new API is simpler to
implement because many methods can be inherited from "parent backends".  It is
recommended if back-compatibility for Matplotlib < 3.6 is not a concern.
However, the old API remains supported.

Fundamentally, a backend module needs to provide information to `.pyplot`, so
that

1. `.pyplot.figure()` can create a new `.Figure` instance and associate it with
   an instance of a backend-provided canvas class, itself hosted in an instance
   of a backend-provided manager class.
2. `.pyplot.show()` can show all figures and start the GUI event loop (if any).

To do so, the backend module must define a ``backend_module.FigureCanvas``
subclass of `.FigureCanvasBase`.  In the canvas-based API, this is the only
strict requirement for backend modules.  The function-based API additionally
requires many module-level functions to be defined.

Canvas-based API (Matplotlib >= 3.6)
------------------------------------

1. **Creating a figure**: `.pyplot.figure()` calls
   ``figure = Figure(); FigureCanvas.new_manager(figure, num)``
   (``new_manager`` is a classmethod) to instantiate a canvas and a manager and
   set up the ``figure.canvas`` and ``figure.canvas.manager`` attributes.
   Figure unpickling uses the same approach, but replaces the newly
   instantiated ``Figure()`` by the unpickled figure.

   Interactive backends should customize the effect of ``new_manager`` by
   setting the ``FigureCanvas.manager_class`` attribute to the desired manager
   class, and additionally (if the canvas cannot be created before the manager,
   as in the case of the wx backends) by overriding the
   ``FigureManager.create_with_canvas`` classmethod.  (Non-interactive backends
   can normally use a trivial ``FigureManagerBase`` and can therefore skip this
   step.)

   After a new figure is registered with `.pyplot` (either via
   `.pyplot.figure()` or via unpickling), if in interactive mode, `.pyplot`
   will call its canvas' ``draw_idle()`` method, which can be overridden as
   desired.

2. **Showing figures**: `.pyplot.show()` calls
   ``FigureCanvas.manager_class.pyplot_show()`` (a classmethod), forwarding any
   arguments, to start the main event loop.

   By default, ``pyplot_show()`` checks whether there are any ``managers``
   registered with `.pyplot` (exiting early if not), calls ``manager.show()``
   on all such managers, and then, if called with ``block=True`` (or with
   the default ``block=None`` and out of IPython's pylab mode and not in
   interactive mode), calls ``FigureCanvas.manager_class.start_main_loop()``
   (a classmethod) to start the main event loop.  Interactive backends should
   therefore override the ``FigureCanvas.manager_class.start_main_loop``
   classmethod accordingly (or alternatively, they may also directly override
   ``FigureCanvas.manager_class.pyplot_show`` directly).

Function-based API
------------------

1. **Creating a figure**: `.pyplot.figure()` calls
   ``new_figure_manager(num, *args, **kwargs)`` (which also takes care of
   creating the new figure as ``Figure(*args, **kwargs)``); unpickling calls
   ``new_figure_manager_given_figure(num, figure)``.

   Furthermore, in interactive mode, the first draw of the newly registered
   figure can be customized by providing a module-level
   ``draw_if_interactive()`` function.  (In the new canvas-based API, this
   function is not taken into account anymore.)

2. **Showing figures**: `.pyplot.show()` calls a module-level ``show()``
   function, which is typically generated via the ``ShowBase`` class and its
   ``mainloop`` method.

Registering a backend
---------------------

For a new backend to be usable via ``matplotlib.use()`` or IPython
``%matplotlib`` magic command, it must be compatible with one of the three ways
supported by the :class:`~matplotlib.backends.registry.BackendRegistry`:

Built-in
^^^^^^^^

A backend built into Matplotlib must have its name and
``FigureCanvas.required_interactive_framework`` hard-coded in the
:class:`~matplotlib.backends.registry.BackendRegistry`.  If the backend module
is not ``f"matplotlib.backends.backend_{backend_name.lower()}"`` then there
must also be an entry in the ``BackendRegistry._name_to_module``.

module:// syntax
^^^^^^^^^^^^^^^^

Any backend in a separate module (not built into Matplotlib) can be used by
specifying the path to the module in the form ``module://some.backend.module``.
An example is ``module://mplcairo.qt`` for
`mplcairo <https:////github.com/matplotlib/mplcairo>`_.  The backend's
interactive framework will be taken from its
``FigureCanvas.required_interactive_framework``.

Entry point
^^^^^^^^^^^

An external backend module can self-register as a backend using an
``entry point`` in its ``pyproject.toml`` such as the one used by
``matplotlib-inline``:

.. code-block:: toml

    [project.entry-points."matplotlib.backend"]
    inline = "matplotlib_inline.backend_inline"

The backend's interactive framework will be taken from its
``FigureCanvas.required_interactive_framework``.  All entry points are loaded
together but only when first needed, such as when a backend name is not
recognised as a built-in backend, or when
:meth:`~matplotlib.backends.registry.BackendRegistry.list_all` is first called.
