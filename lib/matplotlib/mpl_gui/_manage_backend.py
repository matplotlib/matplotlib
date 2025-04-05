import sys
import logging
import types

from matplotlib import cbook, rcsetup
from matplotlib import rcParams, rcParamsDefault
import matplotlib.backend_bases

from matplotlib.backends import backend_registry


_backend_mod = None

_log = logging.getLogger(__name__)


def current_backend_module():
    """
    Get the currently active backend module, selecting one if needed.

    Returns
    -------
    matplotlib.backend_bases._Backend
    """
    if _backend_mod is None:
        select_gui_toolkit()
    return _backend_mod


def select_gui_toolkit(newbackend=None):
    """
    Select the GUI toolkit to use.

    The argument is case-insensitive.  Switching between GUI toolkits is
    possible only if no event loop for another interactive backend has started.
    Switching to and from non-interactive backends is always possible.

    Parameters
    ----------
    newbackend : Union[str, _Backend]
        The name of the backend to use or a _Backend class to use.

    Returns
    -------
    _Backend
       The backend selected.

    """
    global _backend_mod

    # work-around the sentinel resolution in Matplotlib 😱
    if newbackend is None:
        newbackend = dict.__getitem__(rcParams, "backend")

    if newbackend is rcsetup._auto_backend_sentinel:
        current_framework = cbook._get_running_interactive_framework()
        if (current_framework and
                (backend := backend_registry.backend_for_gui_framework(
                    current_framework))):
            candidates = [backend]
        else:
            candidates = []
        candidates += [
            "macosx", "qtagg", "gtk4agg", "gtk3agg", "tkagg", "wxagg"]

        # Don't try to fallback on the cairo-based backends as they each have
        # an additional dependency (pycairo) over the agg-based backend, and
        # are of worse quality.
        for candidate in candidates:
            try:
                return select_gui_toolkit(candidate)
            except ImportError:
                continue

        else:
            # Switching to Agg should always succeed; if it doesn't, let the
            # exception propagate out.
            return select_gui_toolkit("agg")

    if isinstance(newbackend, str):
        # Backends are implemented as modules, but "inherit" default method
        # implementations from backend_bases._Backend.  This is achieved by
        # creating a "class" that inherits from backend_bases._Backend and whose
        # body is filled with the module's globals.

        backend_name = backend_registry.resolve_gui_or_backend(newbackend)[0]
        mod = backend_registry.load_backend_module(newbackend)
        if hasattr(mod, "Backend"):
            orig_class = mod.Backend

        else:
            class orig_class(matplotlib.backend_bases._Backend):
                locals().update(vars(mod))

                @classmethod
                def mainloop(cls):
                    return mod.Show().mainloop()

        class BackendClass(orig_class):
            @classmethod
            def show_managers(cls, *, managers, block):
                if not managers:
                    return
                for manager in managers:
                    manager.show()  # Emits a warning for non-interactive backend
                    manager.canvas.draw_idle()
                if cls.mainloop is None:
                    return
                if block:
                    try:
                        cls.FigureManager._active_managers = managers
                        cls.mainloop()
                    finally:
                        cls.FigureManager._active_managers = None

        if not hasattr(BackendClass.FigureManager, "_active_managers"):
            BackendClass.FigureManager._active_managers = None
        rc_params_string = newbackend

    else:
        BackendClass = newbackend
        mod_name = f"_backend_mod_{id(BackendClass)}"
        rc_params_string = f"module://{mod_name}"
        mod = types.ModuleType(mod_name)
        mod.Backend = BackendClass
        sys.modules[mod_name] = mod

    canvas_class = mod.FigureCanvas
    required_framework = canvas_class.required_interactive_framework
    if required_framework is not None:
        current_framework = cbook._get_running_interactive_framework()
        if (
            current_framework
            and required_framework
            and current_framework != required_framework
        ):
            raise ImportError(
                "Cannot load backend {!r} which requires the {!r} interactive "
                "framework, as {!r} is currently running".format(
                    newbackend, required_framework, current_framework
                )
            )

    _log.debug(
        "Loaded backend %s version %s.", newbackend, BackendClass.backend_version
    )

    rcParams["backend"] = rcParamsDefault["backend"] = rc_params_string

    # is IPython imported?
    mod_ipython = sys.modules.get("IPython")
    if mod_ipython:
        # if so are we in an IPython session
        ip = mod_ipython.get_ipython()
        if ip:
            # macosx -> osx mapping for the osx backend in ipython
            if required_framework == "macosx":
                required_framework = "osx"
            ip.enable_gui(required_framework)

    # remember to set the global variable
    _backend_mod = BackendClass
    return BackendClass
