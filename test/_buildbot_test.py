"""This script will install matplotlib to a virtual environment to
faciltate testing."""
import shutil, os, sys, glob
from subprocess import Popen, PIPE, STDOUT

from _buildbot_util import check_call

TARGET=os.path.abspath('PYmpl')

if not os.path.exists(TARGET):
    raise RuntimeError('the virtualenv target directory was not found')

TARGET_py = os.path.join(TARGET,'bin','python')
check_call('%s -c "import shutil,matplotlib; x=matplotlib.get_configdir(); shutil.rmtree(x)"'%TARGET_py)


previous_test_images = glob.glob(os.path.join('test','failed-diff-*.png'))
for fname in previous_test_images:
    os.unlink(fname)

check_call('%s run-mpl-test.py --verbose --all --keep-failed'%TARGET_py,
           cwd='test')
