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
    def __init__(self):
        # Built-in backends are those which are included in the Matplotlib repo.
        # A backend with name 'name' is located in the module
        # f'matplotlib.backends.backend_{name.lower()}'

        # The capitalized forms are needed for ipython at present; this may
        # change for later versions.
        self._builtin_interactive = [
            "GTK3Agg", "GTK3Cairo", "GTK4Agg", "GTK4Cairo",
            "MacOSX",
            "nbAgg",
            "QtAgg", "QtCairo", "Qt5Agg", "Qt5Cairo",
            "TkAgg", "TkCairo",
            "WebAgg",
            "WX", "WXAgg", "WXCairo",
        ]
        self._builtin_not_interactive = [
            "agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
        ]
        self._framework_to_backend_mapping = {
            "qt": "qtagg",
            "gtk3": "gtk3agg",
            "gtk4": "gtk4agg",
            "wx": "wxagg",
            "tk": "tkagg",
            "macosx": "macosx",
            "headless": "agg",
        }

    def framework_to_backend(self, interactive_framework):
        return self._framework_to_backend_mapping.get(interactive_framework)

    def list_builtin(self, filter_=None):
        if filter_ == BackendFilter.INTERACTIVE:
            return self._builtin_interactive
        elif filter_ == BackendFilter.INTERACTIVE_NON_WEB:
            return list(filter(lambda x: x.lower() not in ("webagg", "nbagg"),
                               self._builtin_interactive))
        elif filter_ == BackendFilter.NON_INTERACTIVE:
            return self._builtin_not_interactive

        return self._builtin_interactive + self._builtin_not_interactive


# Singleton
backend_registry = BackendRegistry()
