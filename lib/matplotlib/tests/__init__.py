from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import difflib
import os

from matplotlib import rcParams, rcdefaults, use


_multiprocess_can_split_ = True


# Check that the test directories exist
if not os.path.exists(os.path.join(
        os.path.dirname(__file__), 'baseline_images')):
    raise IOError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')


def setup():
    # The baseline images are created in this locale, so we should use
    # it during all of the tests.
    import locale
    import warnings
    from matplotlib.backends import backend_agg, backend_pdf, backend_svg

    try:
        locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, str('English_United States.1252'))
        except locale.Error:
            warnings.warn(
                "Could not set locale to English/United States. "
                "Some date-related tests may fail")

    use('Agg', warn=False)  # use Agg backend for these tests

    # These settings *must* be hardcoded for running the comparison
    # tests and are not necessarily the default values as specified in
    # rcsetup.py
    rcdefaults()  # Start with all defaults
    rcParams['font.family'] = 'Bitstream Vera Sans'
    rcParams['text.hinting'] = False
    rcParams['text.hinting_factor'] = 8

    # Clear the font caches.  Otherwise, the hinting mode can travel
    # from one test to another.
    backend_agg.RendererAgg._fontd.clear()
    backend_pdf.RendererPdf.truetype_font_cache.clear()
    backend_svg.RendererSVG.fontd.clear()


def assert_str_equal(reference_str, test_str,
                     format_str=('String {str1} and {str2} do not '
                                 'match:\n{differences}')):
    """
    Assert the two strings are equal. If not, fail and print their
    diffs using difflib.

    """
    if reference_str != test_str:
        diff = difflib.unified_diff(reference_str.splitlines(1),
                                    test_str.splitlines(1),
                                    'Reference', 'Test result',
                                    '', '', 0)
        raise ValueError(format_str.format(str1=reference_str,
                                           str2=test_str,
                                           differences=''.join(diff)))
