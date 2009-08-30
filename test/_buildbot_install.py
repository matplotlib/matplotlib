"""This script will install matplotlib to a virtual environment to
faciltate testing."""
import shutil, os, sys
from subprocess import Popen, PIPE, STDOUT

from _buildbot_util import check_call

TARGET='PYmpl'

if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

if 1:
    build_path = 'build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

check_call('virtualenv %s'%(TARGET,))
TARGET_py = os.path.join(TARGET,'bin','python')
check_call('%s setup.py install'%TARGET_py)
