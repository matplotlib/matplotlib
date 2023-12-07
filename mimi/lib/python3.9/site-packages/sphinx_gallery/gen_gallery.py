# Author: Óscar Nájera
# License: 3-clause BSD
"""Sphinx-Gallery Generator.

Attaches Sphinx-Gallery to Sphinx in order to generate the galleries
when building the documentation.
"""


import codecs
import copy
from datetime import timedelta, datetime
from difflib import get_close_matches
from importlib import import_module
from pathlib import Path
from textwrap import indent
import re
import os
import pathlib
from xml.sax.saxutils import quoteattr, escape
from itertools import chain

from sphinx.errors import ConfigError, ExtensionError
import sphinx.util
from sphinx.util.console import blue, red, purple, bold
from . import glr_path_static, __version__ as _sg_version
from .utils import _replace_md5, _has_optipng, _has_pypandoc, _has_graphviz
from .backreferences import _finalize_backreferences
from .gen_rst import generate_dir_rst, SPHX_GLR_SIG, _get_memory_base, _get_readme
from .scrapers import _scraper_dict, _reset_dict, _import_matplotlib
from .docs_resolv import embed_code_links
from .downloads import generate_zipfiles
from .sorting import NumberOfCodeLinesSortKey
from .interactive_example import (
    copy_binder_files,
    check_binder_conf,
    check_jupyterlite_conf,
)
from .interactive_example import pre_configure_jupyterlite_sphinx
from .interactive_example import post_configure_jupyterlite_sphinx
from .interactive_example import create_jupyterlite_contents
from .directives import MiniGallery, ImageSg, imagesg_addnode
from .recommender import ExampleRecommender, _write_recommendations

_KNOWN_CSS = (
    "sg_gallery",
    "sg_gallery-binder",
    "sg_gallery-dataframe",
    "sg_gallery-rendered-html",
)


class DefaultResetArgv:
    """Provides default 'reset_argv' callable that returns empty list."""

    def __repr__(self):
        return "DefaultResetArgv"

    def __call__(self, gallery_conf, script_vars):
        """Return empty list."""
        return []


DEFAULT_GALLERY_CONF = {
    "filename_pattern": re.escape(os.sep) + "plot",
    "ignore_pattern": r"__init__\.py",
    "examples_dirs": os.path.join("..", "examples"),
    "example_extensions": {".py"},
    "filetype_parsers": {},
    "notebook_extensions": {".py"},
    "reset_argv": DefaultResetArgv(),
    "subsection_order": None,
    "within_subsection_order": NumberOfCodeLinesSortKey,
    "gallery_dirs": "auto_examples",
    "backreferences_dir": None,
    "doc_module": (),
    "exclude_implicit_doc": set(),
    "reference_url": {},
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"",
    # Build options
    # -------------
    # 'plot_gallery' also accepts strings that evaluate to a bool, e.g. "True",
    # "False", "1", "0" so that they can be easily set via command line
    # switches of sphinx-build
    "plot_gallery": "True",
    "download_all_examples": True,
    "abort_on_example_error": False,
    "only_warn_on_example_error": False,
    "recommender": {"enable": False},
    "failing_examples": {},
    "passing_examples": [],
    "stale_examples": [],  # ones that did not need to be run due to md5sum
    "run_stale_examples": False,
    "expected_failing_examples": set(),
    "thumbnail_size": (400, 280),  # Default CSS does 0.4 scaling (160, 112)
    "min_reported_time": 0,
    "binder": {},
    "jupyterlite": {},
    "promote_jupyter_magic": False,
    "image_scrapers": ("matplotlib",),
    "compress_images": (),
    "reset_modules": ("matplotlib", "seaborn"),
    "reset_modules_order": "before",
    "first_notebook_cell": None,
    "last_notebook_cell": None,
    "notebook_images": False,
    "pypandoc": False,
    "remove_config_comments": False,
    "show_memory": False,
    "show_signature": True,
    "junit": "",
    "log_level": {"backreference_missing": "warning"},
    "inspect_global_variables": True,
    "css": _KNOWN_CSS,
    "matplotlib_animations": False,
    "image_srcset": [],
    "default_thumb_file": None,
    "line_numbers": False,
    "nested_sections": True,
    "prefer_full_module": set(),
    "api_usage_ignore": ".*__.*__",
    "show_api_usage": False,  # if this changes, change write_api_entries, too
    "copyfile_regex": "",
}

logger = sphinx.util.logging.getLogger("sphinx-gallery")


def _bool_eval(x):
    """Evaluate bool only configs, to allow setting via -D on the command line."""
    if isinstance(x, str):
        try:
            x = eval(x)
        except TypeError:
            pass
    return bool(x)


def _update_gallery_conf_exclude_implicit_doc(gallery_conf):
    """Update gallery config exclude_implicit_doc.

    This is separate function for better testability.
    """
    # prepare regex for exclusions from implicit documentation
    exclude_regex = (
        re.compile("|".join(gallery_conf["exclude_implicit_doc"]))
        if gallery_conf["exclude_implicit_doc"]
        else False
    )
    gallery_conf["exclude_implicit_doc_regex"] = exclude_regex


def _update_gallery_conf_builder_inited(
    sphinx_gallery_conf,
    src_dir,
    plot_gallery=True,
    abort_on_example_error=False,
    builder_name="html",
):
    sphinx_gallery_conf.update(plot_gallery=plot_gallery)
    sphinx_gallery_conf.update(abort_on_example_error=abort_on_example_error)
    sphinx_gallery_conf["src_dir"] = src_dir
    # Make it easy to know which builder we're in
    sphinx_gallery_conf["builder_name"] = builder_name


def _fill_gallery_conf_defaults(sphinx_gallery_conf, app=None, check_keys=True):
    """Handle user configs and update default gallery configs."""
    gallery_conf = copy.deepcopy(DEFAULT_GALLERY_CONF)
    options = sorted(gallery_conf)
    extra_keys = sorted(set(sphinx_gallery_conf) - set(options))
    if extra_keys and check_keys:
        msg = "Unknown key(s) in sphinx_gallery_conf:\n"
        for key in extra_keys:
            options = get_close_matches(key, options, cutoff=0.66)
            msg += repr(key)
            if len(options) == 1:
                msg += f", did you mean {options[0]!r}?"
            elif len(options) > 1:
                msg += f", did you mean one of {options!r}?"
            msg += "\n"
        raise ConfigError(msg.strip())
    gallery_conf.update(sphinx_gallery_conf)
    # XXX anything that can only be a bool (rather than str) should probably be
    # evaluated this way as it allows setting via -D on the command line
    for key in (
        "promote_jupyter_magic",
        "run_stale_examples",
    ):
        gallery_conf[key] = _bool_eval(gallery_conf[key])
    gallery_conf["app"] = app

    # Check capture_repr
    capture_repr = gallery_conf["capture_repr"]
    supported_reprs = ["__repr__", "__str__", "_repr_html_"]
    if isinstance(capture_repr, tuple):
        for rep in capture_repr:
            if rep not in supported_reprs:
                raise ConfigError(
                    "All entries in 'capture_repr' must be one "
                    f"of {supported_reprs}, got: {rep}"
                )
    else:
        raise ConfigError(f"'capture_repr' must be a tuple, got: {type(capture_repr)}")
    # Check ignore_repr_types
    if not isinstance(gallery_conf["ignore_repr_types"], str):
        raise ConfigError(
            "'ignore_repr_types' must be a string, got: "
            + type(gallery_conf["ignore_repr_types"])
        )

    # deal with show_memory
    gallery_conf["memory_base"] = 0.0
    if gallery_conf["show_memory"]:
        if not callable(gallery_conf["show_memory"]):  # True-like
            try:
                from memory_profiler import memory_usage  # noqa
            except ImportError:
                logger.warning(
                    "Please install 'memory_profiler' to enable "
                    "peak memory measurements."
                )
                gallery_conf["show_memory"] = False
            else:

                def call_memory(func):
                    mem, out = memory_usage(
                        func, max_usage=True, retval=True, multiprocess=True
                    )
                    try:
                        mem = mem[0]  # old MP always returned a list
                    except TypeError:  # 'float' object is not subscriptable
                        pass
                    return mem, out

                gallery_conf["call_memory"] = call_memory
                gallery_conf["memory_base"] = _get_memory_base(gallery_conf)
        else:
            gallery_conf["call_memory"] = gallery_conf["show_memory"]
    if not gallery_conf["show_memory"]:  # can be set to False above

        def call_memory(func):
            return 0.0, func()

        gallery_conf["call_memory"] = call_memory
    assert callable(gallery_conf["call_memory"])

    # deal with scrapers
    scrapers = gallery_conf["image_scrapers"]
    if not isinstance(scrapers, (tuple, list)):
        scrapers = [scrapers]
    scrapers = list(scrapers)
    for si, scraper in enumerate(scrapers):
        if isinstance(scraper, str):
            if scraper in _scraper_dict:
                scraper = _scraper_dict[scraper]
            else:
                orig_scraper = scraper
                try:
                    scraper = import_module(scraper)
                    scraper = getattr(scraper, "_get_sg_image_scraper")
                    scraper = scraper()
                except Exception as exp:
                    raise ConfigError(
                        f"Unknown image scraper {orig_scraper!r}, got:\n{exp}"
                    )
            scrapers[si] = scraper
        if not callable(scraper):
            raise ConfigError(f"Scraper {scraper!r} was not callable")
    gallery_conf["image_scrapers"] = tuple(scrapers)
    del scrapers
    # Here we try to set up matplotlib but don't raise an error,
    # we will raise an error later when we actually try to use it
    # (if we do so) in scrapers.py.
    # In principle we could look to see if there is a matplotlib scraper
    # in our scrapers list, but this would be backward incompatible with
    # anyone using or relying on our Agg-setting behavior (e.g., for some
    # custom matplotlib SVG scraper as in our docs).
    # Eventually we can make this a config var like matplotlib_agg or something
    # if people need us not to set it to Agg.
    try:
        _import_matplotlib()
    except (ImportError, ValueError):
        pass

    # compress_images
    compress_images = gallery_conf["compress_images"]
    if isinstance(compress_images, str):
        compress_images = [compress_images]
    elif not isinstance(compress_images, (tuple, list)):
        raise ConfigError(
            "compress_images must be a tuple, list, or str, "
            f"got {type(compress_images)}"
        )
    compress_images = list(compress_images)
    allowed_values = ("images", "thumbnails")
    pops = list()
    for ki, kind in enumerate(compress_images):
        if kind not in allowed_values:
            if kind.startswith("-"):
                pops.append(ki)
                continue
            raise ConfigError(
                "All entries in compress_images must be one of "
                f"{allowed_values} or a command-line switch "
                f'starting with "-", got {kind!r}'
            )
    compress_images_args = [compress_images.pop(p) for p in pops[::-1]]
    if len(compress_images) and not _has_optipng():
        logger.warning(
            "optipng binaries not found, PNG %s will not be optimized",
            " and ".join(compress_images),
        )
        compress_images = ()
    gallery_conf["compress_images"] = compress_images
    gallery_conf["compress_images_args"] = compress_images_args

    # deal with resetters
    resetters = gallery_conf["reset_modules"]
    if not isinstance(resetters, (tuple, list)):
        resetters = [resetters]
    resetters = list(resetters)
    for ri, resetter in enumerate(resetters):
        if isinstance(resetter, str):
            if resetter not in _reset_dict:
                raise ConfigError(f"Unknown module resetter named {resetter!r}")
            resetters[ri] = _reset_dict[resetter]
        elif not callable(resetter):
            raise ConfigError(f"Module resetter {resetter!r} was not callable")
    gallery_conf["reset_modules"] = tuple(resetters)

    if not isinstance(gallery_conf["reset_modules_order"], str):
        raise ConfigError(
            "reset_modules_order must be a str, "
            f'got {gallery_conf["reset_modules_order"]!r}'
        )
    if gallery_conf["reset_modules_order"] not in ["before", "after", "both"]:
        raise ConfigError(
            "reset_modules_order must be in"
            "['before', 'after', 'both'], "
            f"got {gallery_conf['reset_modules_order']!r}"
        )

    del resetters

    # Ensure the first cell text is a string if we have it
    first_cell = gallery_conf.get("first_notebook_cell")
    if (not isinstance(first_cell, str)) and (first_cell is not None):
        raise ConfigError(
            "The 'first_notebook_cell' parameter must be type "
            f"str or None, found type {type(first_cell)}"
        )
    # Ensure the last cell text is a string if we have it
    last_cell = gallery_conf.get("last_notebook_cell")
    if (not isinstance(last_cell, str)) and (last_cell is not None):
        raise ConfigError(
            "The 'last_notebook_cell' parameter must be type str"
            f" or None, found type {type(last_cell)}"
        )
    # Check pypandoc
    pypandoc = gallery_conf["pypandoc"]
    if not isinstance(pypandoc, (dict, bool)):
        raise ConfigError(
            "'pypandoc' parameter must be of type bool or dict,"
            f"got: {type(pypandoc)}."
        )
    gallery_conf["pypandoc"] = dict() if pypandoc is True else pypandoc
    has_pypandoc, version = _has_pypandoc()
    if isinstance(gallery_conf["pypandoc"], dict) and has_pypandoc is None:
        logger.warning(
            "'pypandoc' not available. Using Sphinx-Gallery to "
            "convert rst text blocks to markdown for .ipynb files."
        )
        gallery_conf["pypandoc"] = False
    elif isinstance(gallery_conf["pypandoc"], dict):
        logger.info(
            "Using pandoc version: %s to convert rst text blocks to "
            "markdown for .ipynb files",
            version,
        )
    else:
        logger.info(
            "Using Sphinx-Gallery to convert rst text blocks to "
            "markdown for .ipynb files."
        )
    if isinstance(pypandoc, dict):
        accepted_keys = ("extra_args", "filters")
        for key in pypandoc:
            if key not in accepted_keys:
                raise ConfigError(
                    "'pypandoc' only accepts the following key "
                    f"values: {accepted_keys}, got: {key}."
                )

    gallery_conf["titles"] = {}
    # Ensure 'backreferences_dir' is str, pathlib.Path or None
    backref = gallery_conf["backreferences_dir"]
    if (not isinstance(backref, (str, pathlib.Path))) and (backref is not None):
        raise ConfigError(
            "The 'backreferences_dir' parameter must be of type "
            "str, pathlib.Path or None, "
            f"found type {type(backref)}"
        )
    # if 'backreferences_dir' is pathlib.Path, make str for Python <=3.5
    # compatibility
    if isinstance(backref, pathlib.Path):
        gallery_conf["backreferences_dir"] = str(backref)

    # binder
    gallery_conf["binder"] = check_binder_conf(gallery_conf["binder"])

    # jupyterlite
    gallery_conf["jupyterlite"] = check_jupyterlite_conf(
        gallery_conf.get("jupyterlite", {}), app
    )

    if not isinstance(gallery_conf["css"], (list, tuple)):
        raise ConfigError(
            'gallery_conf["css"] must be list or tuple, got ' f'{gallery_conf["css"]!r}'
        )
    for css in gallery_conf["css"]:
        if css not in _KNOWN_CSS:
            raise ConfigError(f"Unknown css {css!r}, must be one of {_KNOWN_CSS!r}")
        if gallery_conf["app"] is not None:  # can be None in testing
            gallery_conf["app"].add_css_file(css + ".css")

    # check API usage
    if not isinstance(gallery_conf["api_usage_ignore"], str):
        raise ConfigError(
            'gallery_conf["api_usage_ignore"] must be str, '
            f'got {type(gallery_conf["api_usage_ignore"])}'
        )

    if (
        not isinstance(gallery_conf["show_api_usage"], bool)
        and gallery_conf["show_api_usage"] != "unused"
    ):
        raise ConfigError(
            'gallery_conf["show_api_usage"] must be True, False or "unused", '
            f'got {gallery_conf["show_api_usage"]}'
        )

    _update_gallery_conf_exclude_implicit_doc(gallery_conf)

    return gallery_conf


def get_subsections(srcdir, examples_dir, gallery_conf, check_for_index=True):
    """Return the list of subsections of a gallery.

    Parameters
    ----------
    srcdir : str
        absolute path to directory containing conf.py
    examples_dir : str
        path to the examples directory relative to conf.py
    gallery_conf : Dict[str, Any]
        Sphinx-Gallery configuration dictionary.
    check_for_index : bool
        only return subfolders with a ReadMe, default True

    Returns
    -------
    out : list
        sorted list of gallery subsection folder names
    """
    sortkey = gallery_conf["subsection_order"]
    subfolders = [subfolder for subfolder in os.listdir(examples_dir)]
    if check_for_index:
        subfolders = [
            subfolder
            for subfolder in subfolders
            if _get_readme(
                os.path.join(examples_dir, subfolder), gallery_conf, raise_error=False
            )
            is not None
        ]
    else:
        # just make sure its a directory
        subfolders = [
            subfolder
            for subfolder in subfolders
            if os.path.isdir(os.path.join(examples_dir, subfolder))
        ]
    base_examples_dir_path = os.path.relpath(examples_dir, srcdir)
    subfolders_with_path = [
        os.path.join(base_examples_dir_path, item) for item in subfolders
    ]
    sorted_subfolders = sorted(subfolders_with_path, key=sortkey)

    return [
        subfolders[i]
        for i in [subfolders_with_path.index(item) for item in sorted_subfolders]
    ]


def _prepare_sphx_glr_dirs(gallery_conf, srcdir):
    """Creates necessary folders for sphinx_gallery files."""
    examples_dirs = gallery_conf["examples_dirs"]
    gallery_dirs = gallery_conf["gallery_dirs"]

    if not isinstance(examples_dirs, list):
        examples_dirs = [examples_dirs]

    if not isinstance(gallery_dirs, list):
        gallery_dirs = [gallery_dirs]

    if bool(gallery_conf["backreferences_dir"]):
        backreferences_dir = os.path.join(srcdir, gallery_conf["backreferences_dir"])
        if not os.path.exists(backreferences_dir):
            os.makedirs(backreferences_dir)

    return list(zip(examples_dirs, gallery_dirs))


def _format_toctree(items, includehidden=False):
    """Format a toc tree."""
    st = """
.. toctree::
   :hidden:"""
    if includehidden:
        st += """
   :includehidden:
"""
    st += """

   {}\n""".format(
        "\n   ".join(items)
    )

    st += "\n"

    return st


def generate_gallery_rst(app):
    """Generate the Main examples gallery reStructuredText.

    Start the Sphinx-Gallery configuration and recursively scan the examples
    directories in order to populate the examples gallery.

    We create a 2-level nested structure by iterating through every
    sibling folder of the current index file.
    In each of these folders, we look for a section index file,
    for which we generate a toctree pointing to sibling scripts.
    Then, we append the content of this section index file
    to the current index file,
    after we remove toctree (to keep a clean nested structure)
    and sphinx tags (to prevent tag duplication)
    Eventually, we create a toctree in the current index file
    which points to section index files.
    """
    logger.info("generating gallery...", color="white")
    gallery_conf = app.config.sphinx_gallery_conf

    seen_backrefs = set()

    costs = []
    workdirs = _prepare_sphx_glr_dirs(gallery_conf, app.builder.srcdir)

    # Check for duplicate filenames to make sure linking works as expected
    examples_dirs = [ex_dir for ex_dir, _ in workdirs]
    files = collect_gallery_files(examples_dirs, gallery_conf)
    check_duplicate_filenames(files)
    check_spaces_in_filenames(files)

    for examples_dir, gallery_dir in workdirs:
        examples_dir_abs_path = os.path.join(app.builder.srcdir, examples_dir)
        gallery_dir_abs_path = os.path.join(app.builder.srcdir, gallery_dir)

        # Create section rst files and fetch content which will
        # be added to current index file. This only includes content
        # from files located in the root folder of the current gallery
        # (ie not in subfolders)
        (
            _,
            this_content,
            this_costs,
            this_toctree_items,
        ) = generate_dir_rst(
            examples_dir_abs_path,
            gallery_dir_abs_path,
            gallery_conf,
            seen_backrefs,
            include_toctree=False,
        )

        has_readme = this_content is not None
        costs += this_costs
        write_computation_times(gallery_conf, gallery_dir_abs_path, this_costs)

        # We create an index.rst with all examples
        # (this will overwrite the rst file generated by the previous call
        # to generate_dir_rst)

        if this_content:
            # :orphan: to suppress "not included in TOCTREE" sphinx warnings
            indexst = ":orphan:\n\n" + this_content
        else:
            # we are not going to use the index.rst.new that gets made here,
            # but go through the motions to run through all the subsections...
            indexst = "Never used!"

        # Write toctree with gallery items from gallery root folder
        if len(this_toctree_items) > 0:
            this_toctree = _format_toctree(this_toctree_items)
            indexst += this_toctree

        # list all paths to subsection index files in this array
        subsection_index_files = []
        subsecs = get_subsections(
            app.builder.srcdir,
            examples_dir_abs_path,
            gallery_conf,
            check_for_index=has_readme,
        )
        for subsection in subsecs:
            src_dir = os.path.join(examples_dir_abs_path, subsection)
            target_dir = os.path.join(gallery_dir_abs_path, subsection)
            subsection_index_files.append(
                "/".join(["", gallery_dir, subsection, "index.rst"]).replace(
                    os.sep, "/"
                )  # fwd slashes needed in rst
            )

            (
                subsection_index_path,
                subsection_index_content,
                subsection_costs,
                subsection_toctree_filenames,
            ) = generate_dir_rst(src_dir, target_dir, gallery_conf, seen_backrefs)

            if subsection_index_content:
                # Filter out tags from subsection content
                # to prevent tag duplication across the documentation
                tag_regex = r"^\.\.(\s+)\_(.+)\:(\s*)$"
                subsection_index_content = "\n".join(
                    [
                        line
                        for line in subsection_index_content.splitlines()
                        if re.match(tag_regex, line) is None
                    ]
                    + [""]
                )

                indexst += subsection_index_content
                has_readme_subsection = True
            else:
                has_readme_subsection = False

            # Write subsection toctree in main file only if
            # nested_sections is False or None, and
            # toctree filenames were generated for the subsection.
            if not gallery_conf["nested_sections"]:
                if len(subsection_toctree_filenames) > 0:
                    subsection_index_toctree = _format_toctree(
                        subsection_toctree_filenames
                    )
                    indexst += subsection_index_toctree
            # Otherwise, a new index.rst.new file should
            # have been created and it needs to be parsed
            elif has_readme_subsection:
                _replace_md5(subsection_index_path, mode="t")

            costs += subsection_costs
            write_computation_times(gallery_conf, target_dir, subsection_costs)

        # Build recommendation system
        if gallery_conf["recommender"]["enable"]:
            try:
                import numpy as np  # noqa: F401
            except ImportError:
                raise ConfigError("gallery_conf['recommender'] requires numpy")

            recommender_params = copy.deepcopy(gallery_conf["recommender"])
            recommender_params.pop("enable")
            recommender_params.pop("rubric_header", None)
            recommender = ExampleRecommender(**recommender_params)

            gallery_py_files = []
            # root and subsection directories containing python examples
            gallery_directories = [gallery_dir_abs_path] + subsecs
            for current_dir in gallery_directories:
                src_dir = os.path.join(gallery_dir_abs_path, current_dir)
                # sort python files to have a deterministic input across call
                py_files = sorted(
                    [
                        fname
                        for fname in Path(src_dir).iterdir()
                        if fname.suffix == ".py"
                    ],
                    key=gallery_conf["within_subsection_order"](src_dir),
                )
                gallery_py_files.append(
                    [os.path.join(src_dir, fname) for fname in py_files]
                )
            # flatten the list of list
            gallery_py_files = list(chain.from_iterable(gallery_py_files))

            recommender.fit(gallery_py_files)
            for fname in gallery_py_files:
                _write_recommendations(recommender, fname, gallery_conf)

        # generate toctree with subsections
        if gallery_conf["nested_sections"] is True:
            subsections_toctree = _format_toctree(
                subsection_index_files, includehidden=True
            )

            # add toctree to file only if there are subsections
            if len(subsection_index_files) > 0:
                indexst += subsections_toctree

        if gallery_conf["download_all_examples"]:
            download_fhindex = generate_zipfiles(
                gallery_dir_abs_path, app.builder.srcdir, gallery_conf
            )
            indexst += download_fhindex

        if app.config.sphinx_gallery_conf["show_signature"]:
            indexst += SPHX_GLR_SIG

        if has_readme:
            index_rst_new = os.path.join(gallery_dir_abs_path, "index.rst.new")
            with codecs.open(index_rst_new, "w", encoding="utf-8") as fhindex:
                fhindex.write(indexst)
            _replace_md5(index_rst_new, mode="t")

    # Write a single global sg_execution_times
    write_computation_times(gallery_conf, None, costs)

    if gallery_conf["show_api_usage"] is not False:
        _init_api_usage(app.builder.srcdir)
    _finalize_backreferences(seen_backrefs, gallery_conf)

    if gallery_conf["plot_gallery"]:
        logger.info("computation time summary:", color="white")
        lines, lens = _format_for_writing(
            costs, src_dir=gallery_conf["src_dir"], kind="console"
        )
        for name, t, m in lines:
            text = (f"    - {name}:   ").ljust(lens[0] + 10)
            if t is None:
                text += "(not run)"
                logger.info(text)
            else:
                t_float = float(t.split()[0])
                if t_float >= gallery_conf["min_reported_time"]:
                    text += t.rjust(lens[1]) + "   " + m.rjust(lens[2])
                    logger.info(text)
        # Also create a junit.xml file, useful e.g. on CircleCI
        write_junit_xml(gallery_conf, app.builder.outdir, costs)


SPHX_GLR_ORPHAN = """
:orphan:

.. _{0}:

"""

SPHX_GLR_COMP_TIMES = (
    SPHX_GLR_ORPHAN
    + """
Computation times
=================
"""
)


def _sec_to_readable(t):
    """Convert a number of seconds to a more readable representation."""
    # This will only work for < 1 day execution time
    # And we reserve 2 digits for minutes because presumably
    # there aren't many > 99 minute scripts, but occasionally some
    # > 9 minute ones
    t = datetime(1, 1, 1) + timedelta(seconds=t)
    t = "{:02d}:{:02d}.{:03d}".format(
        t.hour * 60 + t.minute, t.second, int(round(t.microsecond / 1000.0))
    )
    return t


def _cost_key(cost):
    """Cost sorting function."""
    # sort by descending computation time, descending memory, alphabetical name
    return (-cost["t"], -cost["mem"], cost["src_file"])


def _format_for_writing(costs, *, src_dir, kind="rst"):
    """Provide formatted computation summary text.

    Parameters
    ----------
    costs: List[Dict]
        List of dicts of computation costs and paths, see gen_rst.py for details.
    src_dir : pathlib.Path
        The Sphinx source directory.
    kind: 'rst', 'rst-full' or 'console', default='rst'
        Format for printing to 'console' or for writing `sg_execution_times.rst' ('rst'
        for single galleries and 'rst-full' for all galleries).

    Returns
    -------
    lines: List[List[str]]
        Formatted computation text for each example, of format:
        [example_file, time_elapsed, memory_used]

    lens: List[int]
        Character length of each string in `lines`.
    """
    lines = list()
    for cost in sorted(costs, key=_cost_key):
        src_file = cost["src_file"]
        rel_path = os.path.relpath(src_file, src_dir)
        if kind in ("rst", "rst-full"):  # like in sg_execution_times
            target_dir_clean = os.path.relpath(cost["target_dir"], src_dir).replace(
                os.path.sep, "_"
            )
            paren = rel_path if kind == "rst-full" else os.path.basename(src_file)
            name = ":ref:`sphx_glr_{0}_{1}` (``{2}``)".format(
                target_dir_clean, os.path.basename(src_file), paren
            )
            t = _sec_to_readable(cost["t"])
        else:  # like in generate_gallery
            assert kind == "console"
            name = rel_path
            t = f'{cost["t"]:0.2f} sec'
        m = f'{cost["mem"]:.1f} MB'
        lines.append([name, t, m])
    lens = [max(x) for x in zip(*[[len(item) for item in cost] for cost in lines])]
    return lines, lens


def write_computation_times(gallery_conf, target_dir, costs):
    """Write computation times to `sg_execution_times.rst`.

    Parameters
    ----------
    gallery_conf : Dict[str, Any]
        Sphinx-Gallery configuration dictionary.
    target_dir : str | None
        Path to directory where example python source file are.
    costs: List[Dict]
        List of dicts of computation costs and paths, see gen_rst.py for details.
    """
    total_time = sum(cost["t"] for cost in costs)
    if target_dir is None:  # all galleries together
        out_dir = gallery_conf["src_dir"]
        where = "all galleries"
        kind = "rst-full"
        ref_extra = ""
    else:  # a single gallery
        out_dir = target_dir
        where = os.path.relpath(target_dir, gallery_conf["src_dir"])
        kind = "rst"
        ref_extra = f'{where.replace(os.path.sep, "_")}_'
    new_ref = f"sphx_glr_{ref_extra}sg_execution_times"
    out_file = Path(out_dir) / "sg_execution_times.rst"
    if out_file.is_file() and total_time == 0:  # a re-run
        return
    with out_file.open("w", encoding="utf-8") as fid:
        fid.write(SPHX_GLR_COMP_TIMES.format(new_ref))
        fid.write(
            f"**{_sec_to_readable(total_time)}** total execution time for "
            f"{len(costs)} file{'s' if len(costs) != 1 else ''} **from {where}**:\n\n"
        )
        lines, lens = _format_for_writing(
            costs,
            src_dir=gallery_conf["src_dir"],
            kind=kind,
        )
        del costs
        # https://datatables.net/examples/styling/bootstrap5.html
        fid.write(  # put it in a container to make the scoped style work
            """\
.. container::

  .. raw:: html

    <style scoped>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet" />
    </style>
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script type="text/javascript" class="init">
    $(document).ready( function () {
        $('table.sg-datatable').DataTable({order: [[1, 'desc']]});
    } );
    </script>

  .. list-table::
   :header-rows: 1
   :class: table table-striped sg-datatable

   * - Example
     - Time
     - Mem (MB)
"""  # noqa: E501
        )
        # Need at least one entry or Sphinx complains
        for ex, t, mb in lines or [["N/A", "N/A", "N/A"]]:
            fid.write(
                f"""\
   * - {ex}
     - {t}
     - {mb.rsplit(maxsplit=1)[0]}
"""
            )  # remove the "MB" from the right


def write_api_entries(app, what, name, obj, options, lines):
    """Write api entries to `_sg_api_entries` configuration.

    To connect to `autodoc-process-docstring` event.

    Parameters
    ----------
    app :
        The Sphinx application object.
    what: str
        The type of the object which the docstring belongs to. One of
        "module", "class", "exception", "function", "method", "attribute".
    name :
        The fully qualified name of the object.
    obj :
        The object itself.
    options :
        The options given to the directive: an object with attributes inherited_members,
        undoc_members, show_inheritance and no-index that are true if the flag option of
        same name was given to the auto directive.
    lines :
        The lines of the docstring, see above.
    """
    if app.config.sphinx_gallery_conf["show_api_usage"] is False:
        return
    if "_sg_api_entries" not in app.config.sphinx_gallery_conf:
        app.config.sphinx_gallery_conf["_sg_api_entries"] = dict()
    if what not in app.config.sphinx_gallery_conf["_sg_api_entries"]:
        app.config.sphinx_gallery_conf["_sg_api_entries"][what] = set()
    app.config.sphinx_gallery_conf["_sg_api_entries"][what].add(name)


def _init_api_usage(gallery_dir):
    with codecs.open(
        os.path.join(gallery_dir, "sg_api_usage.rst"), "w", encoding="utf-8"
    ):
        pass


# Colors from https://personal.sron.nl/~pault/data/colourschemes.pdf
# 3 Diverging Colour Schemes, Figure 12, plus alpha=AA
API_COLORS = dict(
    edge="#00000080",  # gray (by alpha)
    okay="#98CAE180",  # blue
    bad_1="#FEDA8B80",  # yellow
    bad_2="#F67E4B80",  # orange
    bad_3="#A5002680",  # red
)


def _make_graph(fname, entries, gallery_conf):
    """Make a graph of unused and used API entries.

    The used API entries themselves are documented in the list, so
    for the graph, we'll focus on the number of unused API entries
    per modules. Modules with lots of unused entries (11+) will be colored
    red, those with less (6+) will be colored orange, those with only a few
    (1-5) will be colored yellow and those with no unused entries will be
    colored blue.

    The API entries that are used are shown with one graph per module.
    That way you can see the examples that each API entry is used in
    for that module (if this was done for the whole project at once,
    the graph would get too large very large quickly).

    Parameters
    ----------
    fname: str
        Path to '*sg_api_unused.dot' file.
    entries: Dict[str, List] or List[str]
        Used (List) or unused (Dict) API entries.
    gallery_conf : Dict[str, Any]
        Sphinx-Gallery configuration dictionary.
    """
    import graphviz

    dg = graphviz.Digraph(
        filename=fname,
        graph_attr={
            "overlap": "scale",
            "pad": "0.5",
        },
        node_attr={
            "color": API_COLORS["okay"],
            "style": "filled",
            "fontsize": "20",
            "shape": "box",
            "fontname": "Open Sans,Arial",
        },
    )

    if isinstance(entries, list):
        connections = set()
        lut = dict()  # look up table for connections so they don't repeat
        structs = [entry.split(".") for entry in entries]
        for struct in sorted(structs, key=len):
            for level in range(len(struct) - 2):
                if (struct[level], struct[level + 1]) in connections:
                    continue
                connections.add((struct[level], struct[level + 1]))
                node_from = (
                    lut[struct[level]] if struct[level] in lut else struct[level]
                )
                dg.node(node_from)
                node_to = struct[level + 1]
                node_kwargs = dict()
                # count, don't show leaves
                if len(struct) - 3 == level:
                    leaf_count = 0
                    for struct2 in structs:
                        # find structures of the same length as struct
                        if len(struct2) != level + 3:
                            continue
                        # find structures with two entries before
                        # the leaf that are the same as struct
                        if all(
                            [
                                struct2[level2] == struct[level2]
                                for level2 in range(level + 2)
                            ]
                        ):
                            leaf_count += 1
                    node_to += f" ({leaf_count})"
                    lut[struct[level + 1]] = node_to
                    if leaf_count > 10:
                        color_key = "bad_3"
                    elif leaf_count > 5:
                        color_key = "bad_2"
                    else:
                        color_key = "bad_1"
                    node_kwargs["color"] = API_COLORS[color_key]
                dg.node(node_to, **node_kwargs)
                dg.edge(node_from, node_to, color=API_COLORS["edge"])
        # add modules with all API entries
        for module in gallery_conf["_sg_api_entries"]["module"]:
            struct = module.split(".")
            for i in range(len(struct) - 1):
                if struct[i + 1] not in lut:
                    dg.edge(struct[i], struct[i + 1])
    else:
        assert isinstance(entries, dict)
        for entry, refs in entries.items():
            dg.node(entry)
            for ref in refs:
                dg.node(ref, color=API_COLORS["bad_1"])
                dg.edge(entry, ref, color=API_COLORS["edge"])
    dg.save()


def write_api_entry_usage(app, docname, source):
    """Write an html page describing which API entries are used and unused.

    To document and graph only those API entries that are used by
    autodoc, we have to wait for autodoc to finish and hook into the
    ``source-read`` event. This intercepts the text from the rst such
    that it can be modified. Since, we only touched an empty file,
    we have to add 1) a list of all the API entries that are unused
    and a graph of the number of unused API entries per module and 2)
    a list of API entries that are used in examples, each with a sub-list
    of which examples that API entry is used in, and a graph that
    connects all of the API entries in a module to the examples
    that they are used in.

    Parameters
    ----------
    app :
        The Sphinx application object.
    docname :
        Docname of the document currently being parsed.
    source :
        List whose single element is the contents of the source file
    """
    docname = docname or ""  # can be None on Sphinx 7.2
    if docname != "sg_api_usage":
        return
    gallery_conf = app.config.sphinx_gallery_conf
    if gallery_conf["show_api_usage"] is False:
        return
    # since this is done at the gallery directory level (as opposed
    # to in a gallery directory, e.g. auto_examples), it runs last
    # which means that all the api entries will be in gallery_conf

    # Always write at least the title
    source[0] = SPHX_GLR_ORPHAN.format("sphx_glr_sg_api_usage")
    title = "Unused API Entries"
    source[0] += title + "\n" + "^" * len(title) + "\n\n"
    if (
        "_sg_api_entries" not in gallery_conf
        or gallery_conf["backreferences_dir"] is None
    ):
        source[0] += "No API entries found, not computed.\n\n"
        return
    backreferences_dir = os.path.join(
        gallery_conf["src_dir"], gallery_conf["backreferences_dir"]
    )

    example_files = set.union(
        *[
            gallery_conf["_sg_api_entries"][obj_type]
            for obj_type in ("class", "method", "function")
            if obj_type in gallery_conf["_sg_api_entries"]
        ]
    )

    if len(example_files) == 0:
        source[0] += "No examples run, not computed.\n\n"
        return

    def get_entry_type(entry):
        if entry in gallery_conf["_sg_api_entries"].get("class", []):
            return "class"
        elif entry in gallery_conf["_sg_api_entries"].get("method", []):
            return "meth"
        else:
            assert entry in gallery_conf["_sg_api_entries"]["function"]
            return "func"

    # find used and unused API entries
    unused_api_entries = list()
    used_api_entries = dict()
    for entry in example_files:
        # don't include built-in methods etc.
        if re.match(gallery_conf["api_usage_ignore"], entry) is not None:
            continue
        # check if backreferences empty
        example_fname = os.path.join(backreferences_dir, f"{entry}.examples.new")
        if not os.path.isfile(example_fname):  # use without new
            example_fname = os.path.splitext(example_fname)[0]
        assert os.path.isfile(example_fname)
        if os.path.getsize(example_fname) == 0:
            unused_api_entries.append(entry)
        else:
            used_api_entries[entry] = list()
            with open(example_fname, encoding="utf-8") as fid2:
                for line in fid2:
                    if line.startswith("  :ref:"):
                        example_name = line.split("`")[1]
                        used_api_entries[entry].append(example_name)

    for entry in sorted(unused_api_entries):
        source[0] += f"- :{get_entry_type(entry)}:`{entry}`\n"
    source[0] += "\n\n"

    has_graphviz = _has_graphviz()
    if has_graphviz and unused_api_entries:
        source[0] += (
            ".. graphviz:: ./sg_api_unused.dot\n"
            "    :alt: API unused entries graph\n"
            "    :layout: neato\n\n"
        )

    used_count = len(used_api_entries)
    total_count = used_count + len(unused_api_entries)
    used_percentage = used_count / max(total_count, 1)  # avoid div by zero
    source[0] += (
        "\nAPI entries used: "
        f"{round(used_percentage * 100, 2)}% "
        f"({used_count}/{total_count})\n\n"
    )

    if has_graphviz and unused_api_entries:
        _make_graph(
            os.path.join(app.builder.srcdir, "sg_api_unused.dot"),
            unused_api_entries,
            gallery_conf,
        )

    if gallery_conf["show_api_usage"] is True and used_api_entries:
        title = "Used API Entries"
        source[0] += title + "\n" + "^" * len(title) + "\n\n"
        for entry in sorted(used_api_entries):
            source[0] += f"- :{get_entry_type(entry)}:`{entry}`\n\n"
            for ref in used_api_entries[entry]:
                source[0] += f"  - :ref:`{ref}`\n"
            source[0] += "\n\n"

        if has_graphviz:
            used_modules = {entry.split(".")[0] for entry in used_api_entries}
            for module in sorted(used_modules):
                source[0] += (
                    f"{module}\n" + "^" * len(module) + "\n\n"
                    f".. graphviz:: ./{module}_sg_api_used.dot\n"
                    f"    :alt: {module} usage graph\n"
                    "    :layout: neato\n\n"
                )

            for module in used_modules:
                logger.info("Making API usage graph for %s", module)
                # select and format entries for this module
                entries = dict()
                for entry, ref in used_api_entries.items():
                    if entry.split(".")[0] == module:
                        entry = entry.replace("sphx_glr_", "")
                        # remove prefix
                        for target_dir in gallery_conf["gallery_dirs"]:
                            if entry.startswith(target_dir):
                                entry = entry[len(target_dir) + 1 :]
                _make_graph(
                    os.path.join(app.builder.srcdir, f"{module}_sg_api_used.dot"),
                    entries,
                    gallery_conf,
                )


def clean_api_usage_files(app, exception):
    """Remove api usage .dot files.

    To connect to 'build-finished' event.
    """
    if os.path.isfile(os.path.join(app.builder.srcdir, "sg_api_usage.rst")):
        os.remove(os.path.join(app.builder.srcdir, "sg_api_usage.rst"))
    if os.path.isfile(os.path.join(app.builder.srcdir, "sg_api_unused.dot")):
        os.remove(os.path.join(app.builder.srcdir, "sg_api_unused.dot"))
    for file in os.listdir(app.builder.srcdir):
        if "sg_api_used.dot" in file:
            os.remove(os.path.join(app.builder.srcdir, file))


def write_junit_xml(gallery_conf, target_dir, costs):
    """Write JUnit XML file of example run times, successes, and failures.

    Parameters
    ----------
    gallery_conf : Dict[str, Any]
        Sphinx-Gallery configuration dictionary.
    target_dir : Union[str, pathlib.Path]
        Build directory.
    costs: List[Tuple[Tuple[float], str]]
        List of dicts of computation costs and paths, see gen_rst.py for details.
    """
    if not gallery_conf["junit"] or not gallery_conf["plot_gallery"]:
        return
    failing_as_expected, failing_unexpectedly, passing_unexpectedly = _parse_failures(
        gallery_conf
    )
    n_tests = 0
    n_failures = 0
    n_skips = 0
    elapsed = 0.0
    src_dir = gallery_conf["src_dir"]
    output = ""
    for cost in costs:
        t, fname = cost["t"], cost["src_file"]
        if not any(
            fname in x
            for x in (
                gallery_conf["passing_examples"],
                failing_unexpectedly,
                failing_as_expected,
                passing_unexpectedly,
            )
        ):
            continue  # not subselected by our regex
        title = gallery_conf["titles"][fname]
        output += (
            '<testcase classname={!s} file={!s} line="1" '
            'name={!s} time="{!r}">'.format(
                quoteattr(os.path.splitext(os.path.basename(fname))[0]),
                quoteattr(os.path.relpath(fname, src_dir)),
                quoteattr(title),
                t,
            )
        )
        if fname in failing_as_expected:
            output += '<skipped message="expected example failure"></skipped>'
            n_skips += 1
        elif fname in failing_unexpectedly or fname in passing_unexpectedly:
            if fname in failing_unexpectedly:
                traceback = gallery_conf["failing_examples"][fname]
            else:  # fname in passing_unexpectedly
                traceback = "Passed even though it was marked to fail"
            n_failures += 1
            output += "<failure message={!s}>{!s}</failure>".format(
                quoteattr(traceback.splitlines()[-1].strip()), escape(traceback)
            )
        output += "</testcase>"
        n_tests += 1
        elapsed += t
    output += "</testsuite>"
    output = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuite errors="0" failures="{}" name="sphinx-gallery" '
        'skipped="{}" tests="{}" time="{}">'.format(
            n_failures, n_skips, n_tests, elapsed
        )
    ) + output
    # Actually write it
    fname = os.path.normpath(os.path.join(target_dir, gallery_conf["junit"]))
    junit_dir = os.path.dirname(fname)
    if not os.path.isdir(junit_dir):
        os.makedirs(junit_dir)
    with codecs.open(fname, "w", encoding="utf-8") as fid:
        fid.write(output)


def touch_empty_backreferences(app, what, name, obj, options, lines):
    """Generate empty back-reference example files.

    This avoids inclusion errors/warnings if there are no gallery
    examples for a class / module that is being parsed by autodoc.
    """
    if not bool(app.config.sphinx_gallery_conf["backreferences_dir"]):
        return

    examples_path = os.path.join(
        app.srcdir,
        app.config.sphinx_gallery_conf["backreferences_dir"],
        f"{name}.examples",
    )

    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, "w").close()


def _expected_failing_examples(gallery_conf):
    return {
        os.path.normpath(os.path.join(gallery_conf["src_dir"], path))
        for path in gallery_conf["expected_failing_examples"]
    }


def _parse_failures(gallery_conf):
    """Split the failures."""
    failing_examples = set(gallery_conf["failing_examples"].keys())
    expected_failing_examples = _expected_failing_examples(gallery_conf)
    failing_as_expected = failing_examples.intersection(expected_failing_examples)
    failing_unexpectedly = failing_examples.difference(expected_failing_examples)
    passing_unexpectedly = expected_failing_examples.difference(failing_examples)
    # filter from examples actually run
    passing_unexpectedly = [
        src_file
        for src_file in passing_unexpectedly
        if re.search(gallery_conf["filename_pattern"], src_file)
    ]
    return failing_as_expected, failing_unexpectedly, passing_unexpectedly


def summarize_failing_examples(app, exception):
    """Collects the list of falling examples and prints them with a traceback.

    Raises ValueError if there where failing examples.
    """
    if exception is not None:
        return

    # Under no-plot Examples are not run so nothing to summarize
    if not app.config.sphinx_gallery_conf["plot_gallery"]:
        logger.info(
            'Sphinx-Gallery gallery_conf["plot_gallery"] was '
            "False, so no examples were executed.",
            color="brown",
        )
        return

    gallery_conf = app.config.sphinx_gallery_conf
    failing_as_expected, failing_unexpectedly, passing_unexpectedly = _parse_failures(
        gallery_conf
    )

    idt = "    "
    if failing_as_expected:
        logger.info(bold("Examples failing as expected:"), color="blue")
        for fail_example in failing_as_expected:
            path = os.path.relpath(fail_example, gallery_conf["src_dir"])
            logger.info(
                f"{bold(blue(path))} failed leaving traceback:\n\n"
                f"{indent(gallery_conf['failing_examples'][fail_example], idt)}"
            )

    fail_msgs = []
    if failing_unexpectedly:
        fail_msgs.append(bold(red("Unexpected failing examples:\n")))
        for fail_example in failing_unexpectedly:
            path = os.path.relpath(fail_example, gallery_conf["src_dir"])
            fail_msgs.append(
                f"    {bold(red(path))} failed leaving traceback:\n\n"
                f"{indent(gallery_conf['failing_examples'][fail_example], idt)}"
            )

    if passing_unexpectedly:
        paths = [
            os.path.relpath(p, gallery_conf["src_dir"]) for p in passing_unexpectedly
        ]
        fail_msgs.append(
            bold(red("Examples expected to fail, but not failing:\n\n"))
            + red("\n".join(indent(p, idt) for p in paths))
            + "\n\nPlease remove these examples from "
            + "sphinx_gallery_conf['expected_failing_examples'] "
            + "in your conf.py file."
        )

    # standard message
    n_good = len(gallery_conf["passing_examples"])
    n_tot = len(gallery_conf["failing_examples"]) + n_good
    n_stale = len(gallery_conf["stale_examples"])
    logger.info(
        "\nSphinx-Gallery successfully executed %d out of %d "
        "file%s subselected by:\n\n"
        '    gallery_conf["filename_pattern"] = %r\n'
        '    gallery_conf["ignore_pattern"]   = %r\n'
        "\nafter excluding %d file%s that had previously been run "
        "(based on MD5).\n",
        n_good,
        n_tot,
        "s" if n_tot != 1 else "",
        gallery_conf["filename_pattern"],
        gallery_conf["ignore_pattern"],
        n_stale,
        "s" if n_stale != 1 else "",
        color="brown",
    )

    if fail_msgs:
        fail_message = bold(
            purple(
                "Here is a summary of the problems encountered "
                "when running the examples:\n\n"
                + "\n".join(fail_msgs)
                + "\n"
                + "-" * 79
            )
        )
        if gallery_conf["only_warn_on_example_error"]:
            logger.warning(fail_message)
        else:
            raise ExtensionError(fail_message)


def collect_gallery_files(examples_dirs, gallery_conf):
    """Collect python files from the gallery example directories."""
    files = []
    for example_dir in examples_dirs:
        for root, dirnames, filenames in os.walk(example_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    if re.search(gallery_conf["ignore_pattern"], filename) is None:
                        files.append(os.path.join(root, filename))
    return files


def check_duplicate_filenames(files):
    """Check for duplicate filenames across gallery directories."""
    # Check whether we'll have duplicates
    used_names = set()
    dup_names = list()

    for this_file in files:
        this_fname = os.path.basename(this_file)
        if this_fname in used_names:
            dup_names.append(this_file)
        else:
            used_names.add(this_fname)

    if len(dup_names) > 0:
        logger.warning(
            "Duplicate example file name(s) found. Having duplicate file "
            "names will break some links. "
            "List of files: %s",
            sorted(dup_names),
        )


def check_spaces_in_filenames(files):
    """Check for spaces in filenames across example directories."""
    regex = re.compile(r"[\s]")
    files_with_space = list(filter(regex.search, files))
    if files_with_space:
        logger.warning(
            "Example file name(s) with space(s) found. Having space(s) in "
            "file names will break some links. "
            "List of files: %s",
            sorted(files_with_space),
        )


def get_default_config_value(key):
    """Get default configuration function."""

    def default_getter(conf):
        return conf["sphinx_gallery_conf"].get(key, DEFAULT_GALLERY_CONF[key])

    return default_getter


def fill_gallery_conf_defaults(app, config, check_keys=True):
    """Check the sphinx-gallery config and set its defaults.

    This is called early at config-inited, so that all the rest of the code can
    do things like ``sphinx_gallery_conf['binder']['use_jupyter_lab']``, even
    if the keys have not been set explicitly in conf.py.
    """
    new_sphinx_gallery_conf = _fill_gallery_conf_defaults(
        config.sphinx_gallery_conf, app=app, check_keys=check_keys
    )
    config.sphinx_gallery_conf = new_sphinx_gallery_conf
    config.html_static_path.append(glr_path_static())


def update_gallery_conf_builder_inited(app):
    """Update the the sphinx-gallery config at builder-inited."""
    plot_gallery = _bool_eval(app.builder.config.plot_gallery)
    src_dir = app.builder.srcdir
    abort_on_example_error = _bool_eval(app.builder.config.abort_on_example_error)
    _update_gallery_conf_builder_inited(
        app.config.sphinx_gallery_conf,
        src_dir,
        plot_gallery=plot_gallery,
        abort_on_example_error=abort_on_example_error,
        builder_name=app.builder.name,
    )


def setup(app):
    """Setup Sphinx-Gallery sphinx extension."""
    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
    for key in ["plot_gallery", "abort_on_example_error"]:
        app.add_config_value(key, get_default_config_value(key), "html")

    # Early filling of sphinx_gallery_conf defaults at config-inited
    app.connect("config-inited", fill_gallery_conf_defaults, priority=10)
    # set small priority value, so that pre_configure_jupyterlite_sphinx is
    # called before jupyterlite_sphinx config-inited
    app.connect("config-inited", pre_configure_jupyterlite_sphinx, priority=100)
    # set high priority value, so that post_configure_jupyterlite_sphinx is
    # called after jupyterlite_sphinx config-inited
    app.connect("config-inited", post_configure_jupyterlite_sphinx, priority=900)

    if "sphinx.ext.autodoc" in app.extensions:
        app.connect("autodoc-process-docstring", touch_empty_backreferences)
        app.connect("autodoc-process-docstring", write_api_entries)
        app.connect("source-read", write_api_entry_usage)

    # Add the custom directive
    app.add_directive("minigallery", MiniGallery)
    app.add_directive("image-sg", ImageSg)

    imagesg_addnode(app)

    # Early update of sphinx_gallery_conf at builder-inited
    app.connect("builder-inited", update_gallery_conf_builder_inited, priority=10)
    app.connect("builder-inited", generate_gallery_rst)
    app.connect("build-finished", copy_binder_files)
    app.connect("build-finished", create_jupyterlite_contents)

    app.connect("build-finished", summarize_failing_examples)
    app.connect("build-finished", embed_code_links)
    app.connect("build-finished", clean_api_usage_files)
    metadata = {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": _sg_version,
    }
    return metadata


def setup_module():
    """Hack to stop nosetests running setup() above."""
    pass
