"""This script will install matplotlib to a virtual environment to
faciltate testing."""
import shutil, os, sys
from subprocess import Popen, PIPE, STDOUT
from optparse import OptionParser

from _buildbot_util import check_call

usage = """%prog [options]"""
parser = OptionParser(usage)
parser.add_option('--virtualenv',type='string',default='virtualenv',
                  help='string to invoke virtualenv')
parser.add_option('--easy-install-nose',action='store_true',default=False,
                  help='run "easy_install nose" in the virtualenv')
(options, args) = parser.parse_args()
if len(args)!=0:
    parser.print_help()
    sys.exit(0)

TARGET='PYmpl'

if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

if 1:
    build_path = 'build'
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

check_call('%s %s'%(options.virtualenv,TARGET))
TARGET_py = os.path.join(TARGET,'bin','python')
TARGET_easy_install = os.path.join(TARGET,'bin','easy_install')

if options.easy_install_nose:
    check_call('%s nose'%TARGET_easy_install)
check_call('%s setup.py install'%TARGET_py)
