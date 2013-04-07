import matplotlib

def _initialize():
    backend = matplotlib.get_backend()
    # Import the requested backend into a generic module object
    if backend.startswith('module://'):
        backend_name = backend[9:]
    else:
        backend_name = 'backend_'+backend
        backend_name = backend_name.lower() # until we banish mixed case
        backend_name = 'matplotlib.backends.%s'%backend_name.lower()
    backend_module = __import__(backend_name, globals(), locals(), [backend_name])
    return backend_module

backend = _initialize()

class Timer(backend.Timer):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses backend-specific timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''

del backend, _initialize
