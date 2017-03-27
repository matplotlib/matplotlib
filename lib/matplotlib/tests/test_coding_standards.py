from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fnmatch import fnmatch
import os

from nose.tools import assert_equal
from nose.plugins.skip import SkipTest
from matplotlib.testing.noseclasses import KnownFailureTest

try:
    import pep8
except ImportError:
    HAS_PEP8 = False
else:
    HAS_PEP8 = pep8.__version__ > '1.4.5'

import matplotlib


PEP8_ADDITIONAL_IGNORE = ['E111',
                          'E114',
                          'E115',
                          'E116',
                          'E121',
                          'E122',
                          'E123',
                          'E124',
                          'E125',
                          'E126',
                          'E127',
                          'E128',
                          'E129',
                          'E131',
                          'E265',
                          'E266',
                          'W503']

EXTRA_EXCLUDE_FILE = os.path.join(os.path.dirname(__file__),
                                  '.pep8_test_exclude.txt')


if HAS_PEP8:
    class StandardReportWithExclusions(pep8.StandardReport):
        #: A class attribute to store the exception exclusion file patterns.
        expected_bad_files = []

        #: A class attribute to store the lines of failing tests.
        _global_deferred_print = []

        #: A class attribute to store patterns which have seen exceptions.
        matched_exclusions = set()

        def get_file_results(self):
            # If the file had no errors, return self.file_errors
            # (which will be 0).
            if not self._deferred_print:
                return self.file_errors

            # Iterate over all of the patterns, to find a possible exclusion.
            # If the filename is to be excluded, go ahead and remove the
            # counts that self.error added.
            for pattern in self.expected_bad_files:
                if fnmatch(self.filename, pattern):
                    self.matched_exclusions.add(pattern)
                    # invert the error method's counters.
                    for _, _, code, _, _ in self._deferred_print:
                        self.counters[code] -= 1
                        if self.counters[code] == 0:
                            self.counters.pop(code)
                            self.messages.pop(code)
                        self.file_errors -= 1
                        self.total_errors -= 1
                    return self.file_errors

            # mirror the content of StandardReport, only storing the output to
            # file rather than printing. This could be a feature request for
            # the PEP8 tool.
            self._deferred_print.sort()
            for line_number, offset, code, text, _ in self._deferred_print:
                self._global_deferred_print.append(
                    self._fmt % {'path': self.filename,
                                 'row': self.line_offset + line_number,
                                 'col': offset + 1, 'code': code,
                                 'text': text})
            return self.file_errors


def assert_pep8_conformance(module=matplotlib, exclude_files=None,
                            extra_exclude_file=EXTRA_EXCLUDE_FILE,
                            pep8_additional_ignore=PEP8_ADDITIONAL_IGNORE,
                            dirname=None, expected_bad_files=None,
                            extra_exclude_directories=None):
    """
    Tests the matplotlib codebase against the "pep8" tool.

    Users can add their own excluded files (should files exist in the
    local directory which is not in the repository) by adding a
    ".pep8_test_exclude.txt" file in the same directory as this test.
    The file should be a line separated list of filenames/directories
    as can be passed to the "pep8" tool's exclude list.
    """

    if not HAS_PEP8:
        raise SkipTest('The pep8 tool is required for this test')

    # to get a list of bad files, rather than the specific errors, add
    # "reporter=pep8.FileReport" to the StyleGuide constructor.
    pep8style = pep8.StyleGuide(quiet=False,
                                reporter=StandardReportWithExclusions)
    reporter = pep8style.options.reporter

    if expected_bad_files is not None:
        reporter.expected_bad_files = expected_bad_files

    # Extend the number of PEP8 guidelines which are not checked.
    pep8style.options.ignore = (pep8style.options.ignore +
                                tuple(pep8_additional_ignore))

    # Support for egg shared object wrappers, which are not PEP8 compliant,
    # nor part of the matplotlib repository.
    # DO NOT ADD FILES *IN* THE REPOSITORY TO THIS LIST.
    if exclude_files is not None:
        pep8style.options.exclude.extend(exclude_files)

    # Allow users to add their own exclude list.
    if extra_exclude_file is not None and os.path.exists(extra_exclude_file):
        with open(extra_exclude_file, 'r') as fh:
            extra_exclude = [line.strip() for line in fh if line.strip()]
        pep8style.options.exclude.extend(extra_exclude)

    if extra_exclude_directories:
        pep8style.options.exclude.extend(extra_exclude_directories)

    if dirname is None:
        dirname = os.path.dirname(module.__file__)
    result = pep8style.check_files([dirname])
    if reporter is StandardReportWithExclusions:
        msg = ("Found code syntax errors (and warnings):\n"
               "{0}".format('\n'.join(reporter._global_deferred_print)))
    else:
        msg = "Found code syntax errors (and warnings)."
    assert_equal(result.total_errors, 0, msg)

    # If we've been using the exclusions reporter, check that we didn't
    # exclude files unnecessarily.
    if reporter is StandardReportWithExclusions:
        unexpectedly_good = sorted(set(reporter.expected_bad_files) -
                                   reporter.matched_exclusions)

        if unexpectedly_good:
            raise ValueError('Some exclude patterns were unnecessary as the '
                             'files they pointed to either passed the PEP8 '
                             'tests or do not point to a file:\n  '
                             '{0}'.format('\n  '.join(unexpectedly_good)))


def test_pep8_conformance_installed_files():
    exclude_files = ['_delaunay.py',
                     '_image.py',
                     '_tri.py',
                     '_backend_agg.py',
                     '_tkagg.py',
                     'ft2font.py',
                     '_cntr.py',
                     '_contour.py',
                     '_png.py',
                     '_path.py',
                     'ttconv.py',
                     '_gtkagg.py',
                     '_backend_gdk.py',
                     'pyparsing*',
                     '_qhull.py',
                     '_macosx.py']

    expected_bad_files = ['_cm.py',
                          '_mathtext_data.py',
                          'backend_bases.py',
                          'cbook.py',
                          'collections.py',
                          'dviread.py',
                          'font_manager.py',
                          'fontconfig_pattern.py',
                          'gridspec.py',
                          'legend_handler.py',
                          'mathtext.py',
                          'patheffects.py',
                          'pylab.py',
                          'pyplot.py',
                          'rcsetup.py',
                          'stackplot.py',
                          'texmanager.py',
                          'transforms.py',
                          'type1font.py',
                          'testing/decorators.py',
                          'testing/jpl_units/Duration.py',
                          'testing/jpl_units/Epoch.py',
                          'testing/jpl_units/EpochConverter.py',
                          'testing/jpl_units/StrConverter.py',
                          'testing/jpl_units/UnitDbl.py',
                          'testing/jpl_units/UnitDblConverter.py',
                          'testing/jpl_units/UnitDblFormatter.py',
                          'testing/jpl_units/__init__.py',
                          'tri/triinterpolate.py',
                          'tests/test_axes.py',
                          'tests/test_bbox_tight.py',
                          'tests/test_delaunay.py',
                          'tests/test_dviread.py',
                          'tests/test_image.py',
                          'tests/test_lines.py',
                          'tests/test_mathtext.py',
                          'tests/test_rcparams.py',
                          'tests/test_simplification.py',
                          'tests/test_streamplot.py',
                          'tests/test_subplots.py',
                          'tests/test_triangulation.py',
                          'backends/__init__.py',
                          'backends/backend_agg.py',
                          'backends/backend_cairo.py',
                          'backends/backend_gdk.py',
                          'backends/backend_gtk.py',
                          'backends/backend_gtk3.py',
                          'backends/backend_gtk3cairo.py',
                          'backends/backend_gtkagg.py',
                          'backends/backend_gtkcairo.py',
                          'backends/backend_macosx.py',
                          'backends/backend_pgf.py',
                          'backends/backend_ps.py',
                          'backends/backend_svg.py',
                          'backends/backend_template.py',
                          'backends/backend_tkagg.py',
                          'backends/tkagg.py',
                          'backends/windowing.py',
                          'backends/qt_editor/formlayout.py',
                          'sphinxext/mathmpl.py',
                          'sphinxext/only_directives.py',
                          'sphinxext/plot_directive.py',
                          'projections/__init__.py',
                          'projections/geo.py',
                          'projections/polar.py']
    expected_bad_files = ['*/matplotlib/' + s for s in expected_bad_files]
    assert_pep8_conformance(module=matplotlib,
                            exclude_files=exclude_files,
                            expected_bad_files=expected_bad_files)


def test_pep8_conformance_examples():
    mpldir = os.environ.get('MPL_REPO_DIR', None)
    if mpldir is None:
        # try and guess!
        fp = os.getcwd()
        while len(fp) > 2:
            if os.path.isdir(os.path.join(fp, 'examples')):
                mpldir = fp
                break
            fp, tail = os.path.split(fp)

    if mpldir is None:
        raise KnownFailureTest("can not find the examples, set env "
                               "MPL_REPO_DIR to point to the top-level path "
                               "of the source tree")

    exdir = os.path.join(mpldir, 'examples')
    blacklist = ()
    expected_bad_files = ['*/pylab_examples/table_demo.py',
                          '*/pylab_examples/tricontour_demo.py',
                          '*/pylab_examples/tripcolor_demo.py',
                          '*/pylab_examples/triplot_demo.py',
                          '*/shapes_and_collections/artist_reference.py',
                          '*/pyplots/align_ylabels.py',
                          '*/pyplots/annotate_transform.py',
                          '*/pyplots/pyplot_simple.py',
                          '*/pyplots/annotation_basic.py',
                          '*/pyplots/annotation_polar.py',
                          '*/pyplots/auto_subplots_adjust.py',
                          '*/pyplots/pyplot_two_subplots.py',
                          '*/pyplots/boxplot_demo.py',
                          '*/pyplots/tex_demo.py',
                          '*/pyplots/compound_path_demo.py',
                          '*/pyplots/text_commands.py',
                          '*/pyplots/text_layout.py',
                          '*/pyplots/fig_axes_customize_simple.py',
                          '*/pyplots/whats_new_1_subplot3d.py',
                          '*/pyplots/whats_new_98_4_fancy.py',
                          '*/pyplots/whats_new_98_4_fill_between.py',
                          '*/pyplots/whats_new_98_4_legend.py',
                          '*/pyplots/pyplot_annotate.py',
                          '*/pyplots/whats_new_99_axes_grid.py',
                          '*/pyplots/pyplot_formatstr.py',
                          '*/pyplots/pyplot_mathtext.py',
                          '*/pyplots/whats_new_99_spines.py']
    assert_pep8_conformance(dirname=exdir,
                            extra_exclude_directories=blacklist,
                            pep8_additional_ignore=PEP8_ADDITIONAL_IGNORE +
                            ['E116', 'E501', 'E402'],
                            expected_bad_files=expected_bad_files)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
