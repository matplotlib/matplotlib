"""
Helper functions for testing.
"""
import locale
import logging
import warnings

import matplotlib as mpl
from matplotlib import cbook

_log = logging.getLogger(__name__)


def is_called_from_pytest():
    """Whether we are in a pytest run."""
    return getattr(mpl, '_called_from_pytest', False)


def set_font_settings_for_testing():
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['text.hinting'] = False
    mpl.rcParams['text.hinting_factor'] = 8


def set_reproducibility_for_testing():
    mpl.rcParams['svg.hashsalt'] = 'matplotlib'


def setup():
    # The baseline images are created in this locale, so we should use
    # it during all of the tests.

    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'English_United States.1252')
        except locale.Error:
            _log.warning(
                "Could not set locale to English/United States. "
                "Some date-related tests may fail.")

    mpl.use('Agg', force=True, warn=False)  # use Agg backend for these tests

    with cbook._suppress_matplotlib_deprecation_warning():
        mpl.rcdefaults()  # Start with all defaults

    # These settings *must* be hardcoded for running the comparison tests and
    # are not necessarily the default values as specified in rcsetup.py.
    set_font_settings_for_testing()
    set_reproducibility_for_testing()
