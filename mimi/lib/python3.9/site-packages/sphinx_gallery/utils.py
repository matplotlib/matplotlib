"""Utilities.

Miscellaneous utilities.
"""
# Author: Eric Larson
# License: 3-clause BSD


import hashlib
import os
import re
from shutil import move, copyfile
import subprocess

from sphinx.errors import ExtensionError
import sphinx.util

try:
    from sphinx.util.display import status_iterator  # noqa: F401
except Exception:  # Sphinx < 6
    from sphinx.util import status_iterator  # noqa: F401


logger = sphinx.util.logging.getLogger("sphinx-gallery")


def _get_image():
    try:
        from PIL import Image
    except ImportError as exc:  # capture the error for the modern way
        try:
            import Image
        except ImportError:
            raise ExtensionError(
                "Could not import pillow, which is required "
                f"to rescale images (e.g., for thumbnails): {exc}"
            )
    return Image


def scale_image(in_fname, out_fname, max_width, max_height):
    """Scales image centered in image box using `max_width` and `max_height`.

    The same aspect ratio is retained. If `in_fname` == `out_fname` the image can only
    be scaled down.
    """
    # local import to avoid testing dependency on PIL:
    Image = _get_image()
    img = Image.open(in_fname)
    # XXX someday we should just try img.thumbnail((max_width, max_height)) ...
    width_in, height_in = img.size
    scale_w = max_width / float(width_in)
    scale_h = max_height / float(height_in)

    if height_in * scale_w <= max_height:
        scale = scale_w
    else:
        scale = scale_h

    if scale >= 1.0 and in_fname == out_fname:
        return

    width_sc = int(round(scale * width_in))
    height_sc = int(round(scale * height_in))

    # resize the image using resize; if using .thumbnail and the image is
    # already smaller than max_width, max_height, then this won't scale up
    # at all (maybe could be an option someday...)
    try:  # Pillow 9+
        bicubic = Image.Resampling.BICUBIC
    except Exception:
        bicubic = Image.BICUBIC
    img = img.resize((width_sc, height_sc), bicubic)
    # img.thumbnail((width_sc, height_sc), Image.BICUBIC)
    # width_sc, height_sc = img.size  # necessary if using thumbnail

    # insert centered
    thumb = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))
    pos_insert = ((max_width - width_sc) // 2, (max_height - height_sc) // 2)
    thumb.paste(img, pos_insert)

    try:
        thumb.save(out_fname)
    except OSError:
        # try again, without the alpha channel (e.g., for JPEG)
        thumb.convert("RGB").save(out_fname)


def optipng(fname, args=()):
    """Optimize a PNG in place.

    Parameters
    ----------
    fname : str
        The filename. If it ends with '.png', ``optipng -o7 fname`` will
        be run. If it fails because the ``optipng`` executable is not found
        or optipng fails, the function returns.
    args : tuple
        Extra command-line arguments, such as ``['-o7']``.
    """
    fname = str(fname)
    if fname.endswith(".png"):
        # -o7 because this is what CPython used
        # https://github.com/python/cpython/pull/8032
        try:
            subprocess.check_call(
                ["optipng"] + list(args) + [fname],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, OSError):  # FileNotFoundError
            pass


def _has_optipng():
    try:
        subprocess.check_call(
            ["optipng", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError:  # FileNotFoundError
        return False
    else:
        return True


def get_md5sum(src_file, mode="b"):
    """Returns md5sum of file.

    Parameters
    ----------
    src_file : str
        Filename to get md5sum for.
    mode : 't' or 'b'
        File mode to open file with. When in text mode, universal line endings
        are used to ensure consistency in hashes between platforms.
    """
    if mode == "t":
        kwargs = {"errors": "surrogateescape", "encoding": "utf-8"}
    else:
        kwargs = {}
    with open(src_file, "r" + mode, **kwargs) as src_data:
        src_content = src_data.read()
        if mode == "t":
            src_content = src_content.encode(**kwargs)
        return hashlib.md5(src_content).hexdigest()


def _replace_md5(fname_new, fname_old=None, method="move", mode="b"):
    fname_new = str(fname_new)  # convert possible Path
    assert method in ("move", "copy")
    if fname_old is None:
        assert fname_new.endswith(".new")
        fname_old = os.path.splitext(fname_new)[0]
    replace = True
    if os.path.isfile(fname_old):
        if get_md5sum(fname_old, mode) == get_md5sum(fname_new, mode):
            replace = False
            if method == "move":
                os.remove(fname_new)
        else:
            logger.debug(f"Replacing stale {fname_old} with {fname_new}")
    if replace:
        if method == "move":
            move(fname_new, fname_old)
        else:
            copyfile(fname_new, fname_old)
    assert os.path.isfile(fname_old)


def _has_pypandoc():
    """Check if pypandoc package available."""
    try:
        import pypandoc  # noqa

        # Import error raised only when function called
        version = pypandoc.get_pandoc_version()
    except (ImportError, OSError):
        return None, None
    else:
        return True, version


def _has_graphviz():
    try:
        import graphviz  # noqa F401
    except ImportError as exc:
        logger.info(
            "`graphviz` required for graphical visualization "
            f"but could not be imported, got: {exc}"
        )
        return False
    return True


def _escape_ansi(s):
    """Remove ANSI terminal formatting characters from a string."""
    return re.sub(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]", "", s)
