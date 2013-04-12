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
Timer = backend.Timer
del backend, _initialize
