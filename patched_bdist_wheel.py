"""
Workaround until https://bitbucket.org/pypa/wheel/issues/91/cannot-create-a-file-when-that-file
is fixed...

Only one small patch in the os.name == nt case:
-            basedir_observed = os.path.join(self.data_dir, '..')
+            basedir_observed = os.path.join(self.data_dir, '..', '.')
"""

from wheel.bdist_wheel import bdist_wheel
from distutils.sysconfig import get_python_version
from distutils import log as logger
from wheel.archive import archive_wheelfile
import subprocess
import os
from shutil import rmtree

class patched_bdist_wheel(bdist_wheel):

    def run(self):
        build_scripts = self.reinitialize_command('build_scripts')
        build_scripts.executable = 'python'

        if not self.skip_build:
            self.run_command('build')

        install = self.reinitialize_command('install',
                                            reinit_subcommands=True)
        install.root = self.bdist_dir
        install.compile = False
        install.skip_build = self.skip_build
        install.warn_dir = False

        # A wheel without setuptools scripts is more cross-platform.
        # Use the (undocumented) `no_ep` option to setuptools'
        # install_scripts command to avoid creating entry point scripts.
        install_scripts = self.reinitialize_command('install_scripts')
        install_scripts.no_ep = True

        # Use a custom scheme for the archive, because we have to decide
        # at installation time which scheme to use.
        for key in ('headers', 'scripts', 'data', 'purelib', 'platlib'):
            setattr(install,
                    'install_' + key,
                    os.path.join(self.data_dir, key))

        basedir_observed = ''

        if os.name == 'nt':
            # win32 barfs if any of these are ''; could be '.'?
            # (distutils.command.install:change_roots bug)
            # PATCHED...
            basedir_observed = os.path.join(self.data_dir, '..', '.')
            self.install_libbase = self.install_lib = basedir_observed

        setattr(install,
                'install_purelib' if self.root_is_pure else 'install_platlib',
                basedir_observed)

        logger.info("installing to %s", self.bdist_dir)

        self.run_command('install')

        archive_basename = self.get_archive_basename()

        pseudoinstall_root = os.path.join(self.dist_dir, archive_basename)
        if not self.relative:
            archive_root = self.bdist_dir
        else:
            archive_root = os.path.join(
                self.bdist_dir,
                self._ensure_relative(install.install_base))

        self.set_undefined_options(
            'install_egg_info', ('target', 'egginfo_dir'))
        self.distinfo_dir = os.path.join(self.bdist_dir,
                                         '%s.dist-info' % self.wheel_dist_name)
        self.egg2dist(self.egginfo_dir,
                      self.distinfo_dir)

        self.write_wheelfile(self.distinfo_dir)

        self.write_record(self.bdist_dir, self.distinfo_dir)

        # Make the archive
        if not os.path.exists(self.dist_dir):
            os.makedirs(self.dist_dir)
        wheel_name = archive_wheelfile(pseudoinstall_root, archive_root)

        # Sign the archive
        if 'WHEEL_TOOL' in os.environ:
            subprocess.call([os.environ['WHEEL_TOOL'], 'sign', wheel_name])

        # Add to 'Distribution.dist_files' so that the "upload" command works
        getattr(self.distribution, 'dist_files', []).append(
            ('bdist_wheel', get_python_version(), wheel_name))

        if not self.keep_temp:
            if self.dry_run:
                logger.info('removing %s', self.bdist_dir)
            else:
                rmtree(self.bdist_dir)

