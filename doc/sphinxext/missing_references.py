"""
This is a sphinx extension to freeze your broken reference problems
when using ``nitpicky = True``.

The basic operation is:

1. Add this extension to your ``conf.py`` extensions.
2. Add ``missing_references_write_json = True`` to your ``conf.py``
3. Run sphinx-build. It will generate ``missing-references.json``
    next to your ``conf.py``.
4. Remove ``missing_references_write_json = True`` from your
    ``conf.py`` (or set it to ``False``)
5. Run sphinx-build again, and ``nitpick_ignore`` will
    contain all of the previously failed references.

"""

from collections import defaultdict
import json
import os.path

from docutils.utils import get_source_line
from sphinx.util import logging

logger = logging.getLogger(__name__)


def record_missing_reference_handler(app, env, node, contnode):
    """
    When the sphinx app notices a missing reference, it emits an
    event which calls this function. This function records the missing
    references for analysis at the end of the sphinx build.
    """
    if not app.config.missing_references_enabled:
        # no-op when we are disabled.
        return

    if not hasattr(env, "missing_reference_record"):
        env.missing_reference_record = defaultdict(set)
    record = env.missing_reference_record

    domain = node["refdomain"]
    typ = node["reftype"]
    target = node["reftarget"]
    location = get_location(node, app)

    dtype = "{}:{}".format(domain, typ)

    record[(dtype, target)].add(location)


def get_location(node, app):
    """
    Given a docutils node and a sphinx application, return a string
    representation of the source location of this node.

    Usually, this will be of the form "path/to/file:linenumber". Two
    special values can be emitted, "<external>" for paths which are
    not contained in this source tree (e.g. docstrings included from
    other modules) or "<unknown>", inidcating that the sphinx application
    cannot locate the original source file (usually because an extension
    has injected text into the sphinx parsing engine).
    """
    (path, line) = get_source_line(node)

    if path:

        basepath = os.path.abspath(os.path.join(app.confdir, ".."))
        path = os.path.relpath(path, start=basepath)

        if path.startswith(os.path.pardir):
            path = os.path.join("<external>", os.path.basename(path))

    else:
        path = "<unknown>"

    if line:
        line = str(line)
    else:
        line = ""

    return "%s:%s" % (path, line)


def save_missing_references_handler(app, exc):
    """
    At the end of the sphinx build, either save the missing references to a
    JSON file. Also ensure that all lines of the existing JSON file are still
    necessary.
    """
    if not app.config.missing_references_enabled:
        # no-op when we are disabled.
        return

    json_path = os.path.join(app.confdir,
                             app.config.missing_references_filename)

    records = app.env.missing_reference_record

    # This is a dictionary of {(dtype,target): locations}
    ignored_references = app.env.missing_references_ignored_references

    # Warn about any reference which is no longer missing.
    for (dtype, target), locations in ignored_references.items():
        missing_reference_locations = records.get((dtype, target), [])

        # For each ignored reference location, ensure a missing reference was
        # observed. If it wasn't observed, issue a warning.
        for ignored_refernece_location in locations:
            if ignored_refernece_location not in missing_reference_locations:
                msg = (f"Reference {dtype} {target} for "
                       f"{ignored_refernece_location} can be removed"
                       f" from {app.config.missing_references_filename}."
                        "It is no longer a missing reference in the docs.")
                logger.warning(msg,
                    location=ignored_refernece_location,
                    type='ref',
                    subtype=dtype)

    if app.config.missing_references_write_json:
        _write_missing_references_json(records, json_path)


def _write_missing_references_json(records, json_path):
    """
    Convert ignored references to a format which we can write as JSON

    Convert from ``{(dtype, target): locaitons}`` to
    ``{dtype: {target: locations}}`` since JSON can't serialize tuples.
    """
    transformed_records = defaultdict(dict)

    for (dtype, target), paths in records.items():
        paths = list(paths)
        paths.sort()
        transformed_records[dtype][target] = paths

    with open(json_path, "w") as stream:
        json.dump(transformed_records, stream, indent=2)


def _read_missing_references_json(json_path):
    """
    Convert from the JSON file to the form used internally by this
    extension.

    The JSON file is stored as ``{dtype: {target: [locations,]}}`` since JSON
    can't store dictionary keys which are tuples. We convert this back to
    ``{(dtype,target):[locations]}`` for internal use.

    """
    with open(json_path, "r") as stream:
        data = json.load(stream)

    ignored_references = {}
    for dtype, targets in data.items():
        for target, locations in targets.items():
            ignored_references[(dtype, target)] = locations
    return ignored_references


def prepare_missing_references_handler(app):
    """
    Handler called to initalize this extension once the configuration
    is ready.

    Reads the missing references file and populates ``nitpick_ignore`` if
    appropriate.
    """
    if not app.config.missing_references_enabled:
        # no-op when we are disabled.
        return

    app.env.missing_references_ignored_references = {}

    json_path = os.path.join(app.confdir,
                             app.config.missing_references_filename)
    if not os.path.exists(json_path):
        return

    ignored_references = _read_missing_references_json(json_path)

    app.env.missing_references_ignored_references = ignored_references

    # If we are going to re-write the JSON file, then don't supress missing
    # reference warnings. We want to record a full list of missing references
    # for use later. Otherwise, add all known missing references to
    # ``nitpick_ignore```
    if not app.config.missing_references_write_json:
        app.config.nitpick_ignore.extend(ignored_references.keys())


def setup(app):
    app.add_config_value("missing_references_enabled", True, "env")
    app.add_config_value("missing_references_write_json", False, "env")
    app.add_config_value("missing_references_filename",
                         "missing-references.json", "env")

    app.connect("builder-inited", prepare_missing_references_handler)
    app.connect("missing-reference", record_missing_reference_handler)
    app.connect("build-finished", save_missing_references_handler)
