from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from fnmatch import fnmatch
import os
import sys

from nose.tools import assert_equal
from nose.plugins.skip import SkipTest

try:
    import pep8
except ImportError:
    HAS_PEP8 = False
else:
    HAS_PEP8 = pep8.__version__ > '1.4.5'

import matplotlib


EXTRA_EXCLUDE_FILE = os.path.join(os.path.dirname(__file__),
                                  '.pep8_test_exclude.txt')
EXCLUDE_FILES = ['_delaunay.py',
                 '_image.py',
                 '_tri.py',
                 '_backend_agg.py',
                 '_tkagg.py',
                 'ft2font.py',
                 '_cntr.py',
                 '_png.py',
                 '_path.py',
                 'ttconv.py',
                 '_gtkagg.py',
                 '_backend_gdk.py',
                 'pyparsing*',
                 '_qhull.py',
                 '_macosx.py']

PEP8_ADDITIONAL_IGNORE = ['E111',
                          'E112',
                          'E113',
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
                          'E265']

EXPECTED_BAD_FILES = ['*/matplotlib/__init__.py',
                      '*/matplotlib/_cm.py',
                      '*/matplotlib/_mathtext_data.py',
                      '*/matplotlib/_pylab_helpers.py',
                      '*/matplotlib/afm.py',
                      '*/matplotlib/artist.py',
                      '*/matplotlib/axis.py',
                      '*/matplotlib/backend_bases.py',
                      '*/matplotlib/bezier.py',
                      '*/matplotlib/cbook.py',
                      '*/matplotlib/collections.py',
                      '*/matplotlib/dviread.py',
                      '*/matplotlib/font_manager.py',
                      '*/matplotlib/fontconfig_pattern.py',
                      '*/matplotlib/gridspec.py',
                      '*/matplotlib/legend.py',
                      '*/matplotlib/legend_handler.py',
                      '*/matplotlib/mathtext.py',
                      '*/matplotlib/mlab.py',
                      '*/matplotlib/path.py',
                      '*/matplotlib/patheffects.py',
                      '*/matplotlib/pylab.py',
                      '*/matplotlib/pyplot.py',
                      '*/matplotlib/rcsetup.py',
                      '*/matplotlib/stackplot.py',
                      '*/matplotlib/texmanager.py',
                      '*/matplotlib/transforms.py',
                      '*/matplotlib/type1font.py',
                      '*/matplotlib/widgets.py',
                      '*/matplotlib/testing/decorators.py',
                      '*/matplotlib/testing/image_util.py',
                      '*/matplotlib/testing/noseclasses.py',
                      '*/matplotlib/testing/jpl_units/Duration.py',
                      '*/matplotlib/testing/jpl_units/Epoch.py',
                      '*/matplotlib/testing/jpl_units/EpochConverter.py',
                      '*/matplotlib/testing/jpl_units/StrConverter.py',
                      '*/matplotlib/testing/jpl_units/UnitDbl.py',
                      '*/matplotlib/testing/jpl_units/UnitDblConverter.py',
                      '*/matplotlib/testing/jpl_units/UnitDblFormatter.py',
                      '*/matplotlib/testing/jpl_units/__init__.py',
                      '*/matplotlib/tri/triinterpolate.py',
                      '*/matplotlib/tests/test_axes.py',
                      '*/matplotlib/tests/test_bbox_tight.py',
                      '*/matplotlib/tests/test_delaunay.py',
                      '*/matplotlib/tests/test_dviread.py',
                      '*/matplotlib/tests/test_image.py',
                      '*/matplotlib/tests/test_legend.py',
                      '*/matplotlib/tests/test_lines.py',
                      '*/matplotlib/tests/test_mathtext.py',
                      '*/matplotlib/tests/test_rcparams.py',
                      '*/matplotlib/tests/test_simplification.py',
                      '*/matplotlib/tests/test_spines.py',
                      '*/matplotlib/tests/test_streamplot.py',
                      '*/matplotlib/tests/test_subplots.py',
                      '*/matplotlib/tests/test_tightlayout.py',
                      '*/matplotlib/tests/test_transforms.py',
                      '*/matplotlib/tests/test_triangulation.py',
                      '*/matplotlib/compat/subprocess.py',
                      '*/matplotlib/backends/__init__.py',
                      '*/matplotlib/backends/backend_agg.py',
                      '*/matplotlib/backends/backend_cairo.py',
                      '*/matplotlib/backends/backend_cocoaagg.py',
                      '*/matplotlib/backends/backend_gdk.py',
                      '*/matplotlib/backends/backend_gtk.py',
                      '*/matplotlib/backends/backend_gtk3.py',
                      '*/matplotlib/backends/backend_gtk3cairo.py',
                      '*/matplotlib/backends/backend_gtkagg.py',
                      '*/matplotlib/backends/backend_gtkcairo.py',
                      '*/matplotlib/backends/backend_macosx.py',
                      '*/matplotlib/backends/backend_mixed.py',
                      '*/matplotlib/backends/backend_pgf.py',
                      '*/matplotlib/backends/backend_ps.py',
                      '*/matplotlib/backends/backend_svg.py',
                      '*/matplotlib/backends/backend_template.py',
                      '*/matplotlib/backends/backend_tkagg.py',
                      '*/matplotlib/backends/backend_wx.py',
                      '*/matplotlib/backends/backend_wxagg.py',
                      '*/matplotlib/backends/tkagg.py',
                      '*/matplotlib/backends/windowing.py',
                      '*/matplotlib/backends/qt_editor/formlayout.py',
                      '*/matplotlib/sphinxext/ipython_console_highlighting.py',
                      '*/matplotlib/sphinxext/ipython_directive.py',
                      '*/matplotlib/sphinxext/mathmpl.py',
                      '*/matplotlib/sphinxext/only_directives.py',
                      '*/matplotlib/sphinxext/plot_directive.py',
                      '*/matplotlib/projections/__init__.py',
                      '*/matplotlib/projections/geo.py',
                      '*/matplotlib/projections/polar.py']


if HAS_PEP8:
    class StandardReportWithExclusions(pep8.StandardReport):
        #; A class attribute to store the exception exclusion file patterns.
        expected_bad_files = EXPECTED_BAD_FILES

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


def assert_pep8_conformance(module=matplotlib, exclude_files=EXCLUDE_FILES,
                            extra_exclude_file=EXTRA_EXCLUDE_FILE,
                            pep8_additional_ignore=PEP8_ADDITIONAL_IGNORE):
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

    # Extend the number of PEP8 guidelines which are not checked.
    pep8style.options.ignore = (pep8style.options.ignore +
                                tuple(pep8_additional_ignore))

    # Support for egg shared object wrappers, which are not PEP8 compliant,
    # nor part of the matplotlib repository.
    # DO NOT ADD FILES *IN* THE REPOSITORY TO THIS LIST.
    pep8style.options.exclude.extend(exclude_files)

    # Allow users to add their own exclude list.
    if extra_exclude_file is not None and os.path.exists(extra_exclude_file):
        with open(extra_exclude_file, 'r') as fh:
            extra_exclude = [line.strip() for line in fh if line.strip()]
        pep8style.options.exclude.extend(extra_exclude)

    result = pep8style.check_files([os.path.dirname(module.__file__)])
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
                             '{}'.format('\n  '.join(unexpectedly_good)))


def test_pep8_conformance():
    assert_pep8_conformance()


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
