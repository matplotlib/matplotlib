""" Utility functions for managing resources (ie. images/files etc). """


# Standard library imports.
import inspect, os, sys


def get_path(path):
    """ Returns an absolute path for the specified path.

    'path' can be a string, class or instance.

    """

    if type(path) is not str:
        # Is this a class or an instance?
        if inspect.isclass(path):
            klass = path

        else:
            klass = path.__class__

        # Get the name of the module that the class was loaded from.
        module_name = klass.__module__
        
        # Look the module up.
        module = sys.modules[module_name]

        if module_name == '__main__':
            dirs = [os.path.dirname(sys.argv[0]), os.getcwd()]
            for d in dirs:
                if os.path.exists(d):
                    path = d
                    break
        else:
            # Get the path to the module.
            path = os.path.dirname(module.__file__)

    return path

#### EOF ######################################################################
