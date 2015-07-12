from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

"""
Core functions and attributes for the matplotlib style library:

``use``
    Select style sheet to override the current matplotlib settings.
``context``
    Context manager to use a style sheet temporarily.
``available``
    List available style sheets.
``library``
    A dictionary of style names and matplotlib settings.
"""
import os
import re
import contextlib

import matplotlib as mpl
from matplotlib import cbook
from matplotlib import rc_params_from_file


__all__ = ['use', 'context', 'available', 'library', 'reload_library']


BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
# Users may want multiple library paths, so store a list of paths.
USER_LIBRARY_PATHS = [os.path.join(mpl._get_configdir(), 'stylelib')]
STYLE_EXTENSION = 'mplstyle'
STYLE_FILE_PATTERN = re.compile('([\S]+).%s$' % STYLE_EXTENSION)


def is_style_file(filename):
    """Return True if the filename looks like a style file."""
    return STYLE_FILE_PATTERN.match(filename) is not None


def use(style):
    """Use matplotlib style settings from a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    Parameters
    ----------
    style : str, dict, or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `style.available`.       |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str or dict) applied from first |
        |      | to last in the list.                                        |
        +------+-------------------------------------------------------------+


    """
    if cbook.is_string_like(style) or hasattr(style, 'keys'):
        # If name is a single str or dict, make it a single element list.
        styles = [style]
    else:
        styles = style

    for style in styles:
        if not cbook.is_string_like(style):
            mpl.rcParams.update(style)
            continue
        elif style == 'default':
            mpl.rcdefaults()
            continue

        if style in library:
            mpl.rcParams.update(library[style])
        else:
            try:
                rc = rc_params_from_file(style, use_default_template=False)
                mpl.rcParams.update(rc)
            except IOError:
                msg = ("'%s' not found in the style library and input is "
                       "not a valid URL or path. See `style.available` for "
                       "list of available styles.")
                raise IOError(msg % style)


@contextlib.contextmanager
def context(style, after_reset=False):
    """Context manager for using style settings temporarily.

    Parameters
    ----------
    style : str, dict, or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `style.available`.       |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str or dict) applied from first |
        |      | to last in the list.                                        |
        +------+-------------------------------------------------------------+

    after_reset : bool
        If True, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.
    """
    initial_settings = mpl.rcParams.copy()
    if after_reset:
        mpl.rcdefaults()
    try:
        use(style)
    except:
        # Restore original settings before raising errors during the update.
        mpl.rcParams.update(initial_settings)
        raise
    else:
        yield
    finally:
        mpl.rcParams.update(initial_settings)


def load_base_library():
    """Load style library defined in this package."""
    library = dict()
    library.update(read_style_directory(BASE_LIBRARY_PATH))
    return library


def iter_user_libraries():
    for stylelib_path in USER_LIBRARY_PATHS:
        stylelib_path = os.path.expanduser(stylelib_path)
        if os.path.exists(stylelib_path) and os.path.isdir(stylelib_path):
            yield stylelib_path


def update_user_library(library):
    """Update style library with user-defined rc files"""
    for stylelib_path in iter_user_libraries():
        styles = read_style_directory(stylelib_path)
        update_nested_dict(library, styles)
    return library


def iter_style_files(style_dir):
    """Yield file path and name of styles in the given directory."""
    for path in os.listdir(style_dir):
        filename = os.path.basename(path)
        if is_style_file(filename):
            match = STYLE_FILE_PATTERN.match(filename)
            path = os.path.abspath(os.path.join(style_dir, path))
            yield path, match.groups()[0]


def read_style_directory(style_dir):
    """Return dictionary of styles defined in `style_dir`."""
    styles = dict()
    for path, name in iter_style_files(style_dir):
        styles[name] = rc_params_from_file(path, use_default_template=False)
    return styles


def update_nested_dict(main_dict, new_dict):
    """Update nested dict (only level of nesting) with new values.

    Unlike dict.update, this assumes that the values of the parent dict are
    dicts (or dict-like), so you shouldn't replace the nested dict if it
    already exists. Instead you should update the sub-dict.
    """
    # update named styles specified by user
    for name, rc_dict in six.iteritems(new_dict):
        if name in main_dict:
            main_dict[name].update(rc_dict)
        else:
            main_dict[name] = rc_dict
    return main_dict


# Load style library
# ==================
_base_library = load_base_library()

library = None
available = []


def reload_library():
    """Reload style library."""
    global library, available
    library = update_user_library(_base_library)
    available[:] = library.keys()
reload_library()
