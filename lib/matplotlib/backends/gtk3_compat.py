import matplotlib

error_msg_gtk3 = "Gtk3 backend requires the installation of pygobject or pgi."

# Import the first library that works from the rcParam list
# throw ImportError if none works
for lib in matplotlib.rcParams['backend.gi_preference']:
    try:
        gi = __import__(lib, globals(), locals(), [], 0)
        break
    except ImportError:
        pass
else:
    raise ImportError(error_msg_gtk3)

# Check version
try:
    gi.require_version("Gtk", "3.0")
except AttributeError:
    raise ImportError(
        "pygobject version too old -- it must have require_version")
except ValueError:
    raise ImportError(
        "Gtk3 backend requires the installation of GObject introspection "
        "bindings for Gtk 3")

# cleanly import pkgs to global scope
try:
    pkgs = ['Gtk', 'Gdk', 'GObject', 'GLib']
    name = gi.__name__ + '.repository'
    _temp = __import__(name, globals(), locals(), pkgs, 0)
    globals().update(dict((k, getattr(_temp, k)) for k in pkgs))
except (ImportError, AttributeError):
    raise ImportError(error_msg_gtk3)
