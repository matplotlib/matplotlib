"""
Core functions and attributes for the matplotlib style library:

``use``
    Select style sheet to override the current matplotlib settings.
``available``
    List available style sheets.
``library``
    A dictionary of style names and matplotlib settings.
"""
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


__all__ = ['use', 'available', 'library']


_here = os.path.abspath(os.path.dirname(__file__))
BASE_LIBRARY_PATH = os.path.join(_here, 'stylelib')
# Users may want multiple library paths, so store a list of paths.
USER_LIBRARY_PATHS = [os.path.join('~', '.matplotlib', 'stylelib')]
STYLE_FILE_PATTERN = re.compile('([A-Za-z._-]+).mplrc$')


def use(name):
    """Use matplotlib rc parameters from a pre-defined name or from a file.

    Parameters
    ----------
    name : str or list of str
        Name of style. For list of available styles see `style.available`.
        If given a list, each style is applied from first to last in the list.
    """
    if np.isscalar(name):
        name = [name]
    for s in name:
        plt.rcParams.update(library[s])


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
        match = STYLE_FILE_PATTERN.match(filename)
        if match:
            path = os.path.abspath(os.path.join(style_dir, path))
            yield path, match.groups()[0]


def read_style_directory(style_dir):
    """Return dictionary of styles defined in `style_dir`."""
    styles = dict()
    for path, name in iter_style_files(style_dir):
        styles[name] = mpl.rc_params_in_file(path)
    return styles


def update_nested_dict(main_dict, new_dict):
    """Update nested dict (only level of nesting) with new values.

    Unlike dict.update, this assumes that the values of the parent dict are
    dicts (or dict-like), so you shouldn't replace the nested dict if it
    already exists. Instead you should update the sub-dict.
    """
    # update named styles specified by user
    for name, rc_dict in new_dict.iteritems():
        if name in main_dict:
            # FIXME: This is currently broken because rc_params_from_file fills
            # in all settings so the update overwrites all values.
            main_dict[name].update(rc_dict)
        else:
            main_dict[name] = rc_dict
    return main_dict


# Load style library
# ==================
_base_library = load_base_library()
library = update_user_library(_base_library)
available = library.keys()
