import os
from pathlib import Path
import sys


rootdir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, rootdir)
os.chdir(rootdir)


from setuptools import setup, find_packages
import versioneer


__version__ = versioneer.get_version()


if "sdist" in sys.argv:
    sys.exit("Only wheels can be generated.")
setup(
    name="matplotlib.tests",
    version=__version__,
    package_dir={"": "lib"},
    packages=find_packages("lib", include=["*.tests"]),
    include_package_data=True,
    install_requires=["matplotlib=={}".format(__version__)]
)
