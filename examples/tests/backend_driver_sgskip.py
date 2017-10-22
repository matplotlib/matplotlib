"""
==============
Backend Driver
==============

This is used to drive many of the examples across the backends, for
regression testing, and comparing backend efficiency.

You can specify the backends to be tested via the --backends
switch, which takes a space-separated list, e.g.

    python backend_driver_sgskip.py --backends agg ps

would test the agg and ps backends. If --backends is not specified, a
default list of backends will be tested.
"""

import os
import time
import sys
import glob
from argparse import ArgumentParser

import matplotlib.rcsetup as rcsetup
from matplotlib.cbook import Bunch, dedent


all_backends = list(rcsetup.all_backends)  # to leave the original list alone

dirs = dict()
files = dict()
for subdir, _, f in os.walk('../'):
    if subdir in ['../', '../tests']:
        continue
    dirs[subdir[3:]] = subdir
    files[subdir[3:]] = [s for s in f if s.endswith('.py')]

# list of whole directories to not run
excluded_dirs = ['user_interfaces']
# dict of files to not run on any backend
excluded_files = {'units': ['date_support.py'],
                  'widgets': ['lasso_selector_demo.py'],
                  'pylab_examples': ['ginput_demo.py',
                                     'ginput_manual_clabel.py']}
# dict of files to not run on specific backend
failbackend = {'svg': ['tex_demo.py'],
               'agg': ['hyperlinks.py'],
               'pdf': ['hyperlinks.py'],
               'ps': ['hyperlinks.py']}

import subprocess


def run(arglist):
    try:
        ret = subprocess.check_output(arglist)
    except KeyboardInterrupt:
        sys.exit()
    else:
        return ret


def drive(backend, directories, python=[sys.executable], switches=[]):
    exclude = failbackend.get(backend, [])

    # Clear the destination directory for the examples
    path = backend
    if os.path.exists(path):
        for fname in os.listdir(path):
            os.unlink(os.path.join(path, fname))
    else:
        os.mkdir(backend)
    failures = []
    # If no directories specified, test all directories
    if directories is None:
        directories = dirs

    testcases = [os.path.join(dirs[d], fname)
                 for d in directories
                 for fname in files[d]]

    for fullpath in testcases:
        print('\tdriving %-40s' % (fullpath))
        sys.stdout.flush()
        fpath, fname = os.path.split(fullpath)
        fdir = os.path.basename(fpath)

        if fname in exclude:
            print('\tSkipping %s, known to fail on backend: %s' %
                  (fname, backend))
            continue
        if fdir in excluded_dirs:
            print('All tests in %s are configured to be skipped' % fdir)
            continue
        if fdir in excluded_files:
            if fname in excluded_files[fdir]:
                print('Skipping %s' % fname)
                continue

        basename, ext = os.path.splitext(fname)
        outfile = os.path.join(path, basename)
        tmpfile_name = '_tmp_%s.py' % basename
        tmpfile = open(tmpfile_name, 'w')

        for line in open(fullpath):
            line_lstrip = line.lstrip()
            if line_lstrip.startswith("#"):
                tmpfile.write(line)

        tmpfile.writelines((
            'import sys\n',
            'sys.path.append("%s")\n' % fpath.replace('\\', '\\\\'),
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % backend,
            'from pylab import savefig\n',
            'import numpy\n',
            'numpy.seterr(invalid="ignore")\n',
        ))
        for line in open(fullpath):
            strip_lines = ['from __future__ import',
                           'matplotlib.use',
                           'plt.savefig',
                           'plt.show']
            line_lstrip = line.lstrip()
            if any(strip in line for strip in strip_lines):
                continue
            tmpfile.write(line)
        if backend in rcsetup.interactive_bk:
            tmpfile.write('show()')
        else:
            tmpfile.write('\nsavefig(r"%s", dpi=150)' % outfile)

        tmpfile.close()
        start_time = time.time()
        program = [x % {'name': basename} for x in python]
        try:
            ret = run(program + [tmpfile_name] + switches)
        except subprocess.CalledProcessError:
            failures.append(fullpath)
        finally:
            end_time = time.time()
            print("%s %s" % ((end_time - start_time), ret))
            os.remove(tmpfile_name)

    return failures


def parse_options():
    doc = (__doc__ and __doc__.split('\n\n')) or "  "
    parser = ArgumentParser(description=doc[0].strip(),
                            epilog='\n'.join(doc[1:]))

    helpstr = 'Run only the tests in these directories'
    parser.add_argument('-d', '--dirs', '--directories', type=str,
                        dest='dirs', help=helpstr, nargs='+')

    helpstr = ('Run tests only for these backends; list of '
               'one or more of: agg, ps, svg, pdf, template, cairo. Default '
               'is everything except cairo.')
    parser.add_argument('-b', '--backends', type=str, dest='backends',
                        help=helpstr, nargs='+')

    parser.add_argument('--clean', action='store_true', dest='clean',
                        help='Remove result directories, run no tests')

    parser.add_argument('-c', '--coverage', action='store_true',
                        dest='coverage', help='Run in coverage.py')

    parser.add_argument('-v', '--valgrind', action='store_true',
                        dest='valgrind', help='Run in valgrind')

    args, switches = parser.parse_known_args()
    if args.backends is None:
        args.backends = ['agg', 'ps', 'svg', 'pdf', 'template']
    return args, switches


if __name__ == '__main__':
    times = {}
    failures = {}
    options, switches = parse_options()

    if options.clean:
        localdirs = [d for d in glob.glob('*') if os.path.isdir(d)]
        all_backends_set = set(all_backends)
        for d in localdirs:
            if d.lower() not in all_backends_set:
                continue
            print('removing %s' % d)
            for fname in glob.glob(os.path.join(d, '*')):
                os.remove(fname)
            os.rmdir(d)
        for fname in glob.glob('_tmp*.py'):
            os.remove(fname)

        print('all clean...')
        raise SystemExit
    if options.coverage:
        python = ['coverage.py', '-x']
    elif options.valgrind:
        python = ['valgrind', '--tool=memcheck', '--leak-check=yes',
                  '--log-file=%(name)s', sys.executable]
    else:
        python = [sys.executable]

    for backend in options.backends:
        print('testing %s %s' % (backend, ' '.join(switches)))
        t0 = time.time()
        failures[backend] = \
            drive(backend, options.dirs, python, switches)
        t1 = time.time()
        times[backend] = (t1 - t0) / 60

    for backend, elapsed in times.items():
        print('Backend %s took %1.2f minutes to complete' % (backend, elapsed))
        failed = failures[backend]
        if failed:
            print('  Failures: %s' % failed)
        if 'template' in times:
            print('\ttemplate ratio %1.3f, template residual %1.3f' % (
                elapsed / times['template'], elapsed - times['template']))
