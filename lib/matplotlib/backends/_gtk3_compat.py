"""
GObject compatibility loader; supports ``gi`` and ``pgi``.

The binding selection rules are as follows:
- if ``gi`` has already been imported, use it; else
- if ``pgi`` has already been imported, use it; else
- if ``gi`` can be imported, use it; else
- if ``pgi`` can be imported, use it; else
- error out.

Thus, to force usage of PGI when both bindings are installed, import it first.
"""

import importlib
import sys

if "gi" in sys.modules:
    import gi
elif "pgi" in sys.modules:
    import pgi as gi
else:
    try:
        import gi
    except ImportError:
        try:
            import pgi as gi
        except ImportError:
            raise ImportError("The GTK3 backends require PyGObject or pgi")

from .backend_cairo import cairo  # noqa
# The following combinations are allowed:
#   gi + pycairo
#   gi + cairocffi
#   pgi + cairocffi
# (pgi doesn't work with pycairo)
# We always try to import cairocffi first so if a check below fails it means
# that cairocffi was unavailable to start with.
if gi.__name__ == "pgi" and cairo.__name__ == "cairo":
    raise ImportError("pgi and pycairo are not compatible")

if gi.__name__ == "pgi" and gi.version_info < (0, 0, 11, 2):
    raise ImportError("The GTK3 backends are incompatible with pgi<0.0.11.2")
try:
    # :raises ValueError: If module/version is already loaded, already
    # required, or unavailable.
    gi.require_version("Gtk", "3.0")
except ValueError as e:
    # in this case we want to re-raise as ImportError so the
    # auto-backend selection logic correctly skips.
    raise ImportError from e

globals().update(
    {name:
     importlib.import_module("{}.repository.{}".format(gi.__name__, name))
     for name in ["GLib", "GObject", "Gtk", "Gdk"]})
