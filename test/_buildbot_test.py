"""This script will install matplotlib to a virtual environment to
faciltate testing."""
import os, glob

from _buildbot_util import check_call

TARGET=os.path.abspath('PYmpl')

if not os.path.exists(TARGET):
    raise RuntimeError('the virtualenv target directory was not found')

TARGET_py = os.path.join(TARGET,'bin','python')
check_call('%s -c "import shutil,matplotlib; x=matplotlib.get_configdir(); shutil.rmtree(x)"'%TARGET_py)

previous_test_images = glob.glob('failed-diff-*.png')
for fname in previous_test_images:
    os.unlink(fname)

check_call('%s -c "import sys, matplotlib; success = matplotlib.test(verbosity=2); sys.exit(not success)"'%TARGET_py)
