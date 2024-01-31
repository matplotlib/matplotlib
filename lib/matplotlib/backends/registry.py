from enum import Enum


class BackendFilter(Enum):
    INTERACTIVE = 0
    INTERACTIVE_NON_WEB = 1
    NON_INTERACTIVE = 2


class BackendRegistry:
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.
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
        return self._GUI_FRAMEWORK_TO_BACKEND_MAPPING.get(framework)

    def list_builtin(self, filter_=None):
        if filter_ == BackendFilter.INTERACTIVE:
            return self._BUILTIN_INTERACTIVE
        elif filter_ == BackendFilter.INTERACTIVE_NON_WEB:
            return list(filter(lambda x: x.lower() not in ("webagg", "nbagg"),
                               self._BUILTIN_INTERACTIVE))
        elif filter_ == BackendFilter.NON_INTERACTIVE:
            return self._BUILTIN_NOT_INTERACTIVE

        return self._BUILTIN_INTERACTIVE + self._BUILTIN_NOT_INTERACTIVE


# Singleton
backend_registry = BackendRegistry()
