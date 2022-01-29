"""
This module provides auto font selection for variable locales
to avoid display mistaken.
See "examples/pyplots/locale_font.py" for usage example.
"""

from locale import getdefaultlocale
from os.path import sep, join
from os import listdir
from sys import stderr

from matplotlib import rcParams, get_data_path
from matplotlib.font_manager import fontManager
from matplotlib.style import available

"""
Fonts corresponds to each locales.

{ <locale>: [[font_names...], [font_filenames...]] }
"""
locale_font = {
    "zh_CN": [["SimSun", "SimHei", "Songti SC"], ["simsun.ttf", "simhei.ttf", "songti.ttf"]]
}


def _match_font_name(lc):
    """
    Find the first font in locale_font[lc][0] that appears in
    matplotlib.font_manager.fontManager.ttflist

    Parameters
    ----------
    lc : str
        locale

    Returns
    -------
    (int, str)
        returns the index and the font name of the font.
        If no fonts were found, return (None, None)
    """
    available_font_names = [f.name for f in fontManager.ttflist]
    expected_fonts = locale_font[lc][0]
    for i, f in enumerate(expected_fonts):
        if f in available_font_names:
            return i, f
    else:
        return None, None


def _match_font_filename(lc):
    """
    Find the first font file in locale_font[lc][1] that appears in
    <data_path>/fonts/ttf

    Parameters
    ----------
    lc : str
        locale

    Returns
    -------
    (int, str)
        returns the index and the font name of the font.
        If no fonts were found, return (None, None)
    """
    ttf_dir = join(get_data_path(), "fonts", "ttf")
    available_files = listdir(ttf_dir)
    expected_files = locale_font[lc][1]
    for i, file in enumerate(expected_files):
        if file in available_files:
            fontManager.addfont(join(ttf_dir, file))
            f = locale_font[lc][0][i]
            return i, f
    else:
        return None, None


def use_locale_font():
    """
    Set rc parameter "font.family" as the best font,
    where "best" means it has the minimum index among those 
    available on this computer.

    Returns
    -------
    str
        the name of the best font. If no best font, return None
    """
    lc = getdefaultlocale()[0]

    if lc in locale_font.keys():
        i1, f1 = _match_font_name(lc)
        i2, f2 = _match_font_filename(lc)
        if i1 is not None and i2 is not None:
            if i1 <= i2:
                rcParams["font.family"] = f1
                return f1
            else:
                rcParams["font.family"] = f2
                return f2
        elif i1 is not None:
            rcParams["font.family"] = f1
            return f1
        elif i2 is not None:
            rcParams["font.family"] = f2
            return f2
        else:
            stderr.write(f"No supported font for locale {lc}\n. "
                         f"Supported ones are {locale_font[lc][1]}, "
                         f"which you can download and place into "
                         f"{join(get_data_path(), 'fonts', 'ttf')}")
    else:
        stderr.write(f"Locale {lc} is not configured\n")

