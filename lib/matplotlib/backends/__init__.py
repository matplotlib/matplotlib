import sys
import matplotlib

__all__ = ['backend','show','draw_if_interactive','error_msg',
           'new_figure_manager']

interactive_bk     = ['Template','GTK','GTKAgg','FltkAgg','TkAgg','WX','WXAgg']
non_interactive_bk = ['PS','GD','Agg','Paint']
all_backends       = interactive_bk + non_interactive_bk

backend = matplotlib.get_backend()
if backend not in all_backends:
    raise ValueError, 'Unrecognized backend %s' % backend

# Import the requested backend into a generic module object
backend_name = 'backend_'+backend.lower()
backend_mod = __import__('matplotlib.backends.'+backend_name,
                         globals(),locals(),[backend_name])

# Things we pull in from all backends
new_figure_manager = backend_mod.new_figure_manager

# Now define the public API according to the kind of backend in use
if backend in interactive_bk:
    error_msg  = backend_mod.error_msg
    show       = backend_mod.show
    __draw_int = backend_mod.draw_if_interactive

    # wrap draw_if_interactive with a flag which detects if it was called.
    # This allows tools like ipython to properly manage interactive scripts.
    # In python 2.4, this can be cleanly done with a simple decorator.
    def draw_if_interactive():
        draw_if_interactive._called = True
        __draw_int()
    # Flag to store state, so external callers (like ipython) can keep track
    # of draw calls.
    draw_if_interactive._called = False
    draw_if_interactive.__doc__ = __draw_int.__doc__
else:  # non-interactive backends
    def draw_if_interactive():  pass
    def show(): pass
    def error_msg(m):
        print >>sys.stderr, m
        sys.exit()

# Additional imports which only happen for certain backends.  This section
# should probably disappear once all backends are uniform.
if backend=='Paint':
    from backend_paint import error_msg
elif backend in ['WX','WXAgg']:
    Toolbar = backend_mod.Toolbar
    __all__.append('Toolbar')

