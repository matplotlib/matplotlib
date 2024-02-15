from enum import Enum


class BackendFilter(Enum):
    """
    Filter used with :meth:`~matplotlib.backends.registry.BackendRegistry.list_builtin`

    .. versionadded:: 3.9
    """
    INTERACTIVE = 0
    NON_INTERACTIVE = 1


class BackendRegistry:
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.

    All use of ``BackendRegistry`` should be via the singleton instance
    ``backend_registry`` which can be imported from ``matplotlib.backends``.

    .. versionadded:: 3.9
    """
    # Built-in backends are those which are included in the Matplotlib repo.
    # A backend with name 'name' is located in the module
    # f'matplotlib.backends.backend_{name.lower()}'

    # The capitalized forms are needed for ipython at present; this may
    # change for later versions.
    _BUILTIN_INTERACTIVE = [
        "GTK3Agg", "GTK3Cairo", "GTK4Agg", "GTK4Cairo",
        "MacOSX",
        "nbAgg",
        "QtAgg", "QtCairo", "Qt5Agg", "Qt5Cairo",
        "TkAgg", "TkCairo",
        "WebAgg",
        "WX", "WXAgg", "WXCairo",
    ]
    _BUILTIN_NOT_INTERACTIVE = [
        "agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
    ]
    _GUI_FRAMEWORK_TO_BACKEND_MAPPING = {
        "qt": "qtagg",
        "gtk3": "gtk3agg",
        "gtk4": "gtk4agg",
        "wx": "wxagg",
        "tk": "tkagg",
        "macosx": "macosx",
        "headless": "agg",
    }

    def backend_for_gui_framework(self, framework):
        """
        Return the name of the backend corresponding to the specified GUI framework.

        Parameters
        ----------
        framework : str
            GUI framework such as "qt".

        Returns
        -------
        str
            Backend name.
        """
        return self._GUI_FRAMEWORK_TO_BACKEND_MAPPING.get(framework)

    def list_builtin(self, filter_=None):
        """
        Return list of backends that are built into Matplotlib.

        Parameters
        ----------
        filter_ : `~.BackendFilter`, optional
            Filter to apply to returned backends. For example, to return only
            non-interactive backends use `.BackendFilter.NON_INTERACTIVE`.

        Returns
        -------
        list of str
            Backend names.
        """
        if filter_ == BackendFilter.INTERACTIVE:
            return self._BUILTIN_INTERACTIVE
        elif filter_ == BackendFilter.NON_INTERACTIVE:
            return self._BUILTIN_NOT_INTERACTIVE

        return self._BUILTIN_INTERACTIVE + self._BUILTIN_NOT_INTERACTIVE


# Singleton
backend_registry = BackendRegistry()
