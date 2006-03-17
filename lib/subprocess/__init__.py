# matplotlib has copied the subprocess module from Python and made it
# a package. Here we import the subprocess module into the package
# namespace. This way we keep subprocess.py named subprocess.py, but
# it gets installed into its own site-packages/subprocess folder.
from subprocess import *
