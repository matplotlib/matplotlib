try:
    __import__('pkg_resources').declare_namespace(__name__)
    print __name__, __file__
except ImportError:
    pass # must not have setuptools
