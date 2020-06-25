from setuptools import setup, find_packages
import sys
import subprocess

# The version number of the matplotlib_baseline_images should be same as that
# of matplotlib package. Checking the matplotlib version number needs cwd as
# the grandparent directory without changing the directory in the
# sub-wheels/matplotlib-baseline-images/setup.py.
__version__ = subprocess.check_output(
    [sys.executable, "setup.py", "--version"], cwd="../..",
    universal_newlines=True).rstrip("\n")

setup(
    name="matplotlib-baseline-images",
    version=__version__,
    description="Package contains mpl baseline images and mpl toolkit images",
    package_dir={"": "lib"},
    packages=find_packages("lib"),
    include_package_data=True,
#    install_requires=["matplotlib=={}".format(__version__)],
)
