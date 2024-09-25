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
from pathlib import Path

from docutils.utils import get_source_line
from sphinx.util import logging as sphinx_logging

import matplotlib

logger = sphinx_logging.getLogger(__name__)


def get_location(node, app):
    """
    Given a docutils node and a sphinx application, return a string
    representation of the source location of this node.

    Usually, this will be of the form "path/to/file:linenumber". Two
    special values can be emitted, "<external>" for paths which are
    not contained in this source tree (e.g. docstrings included from
    other modules) or "<unknown>", indicating that the sphinx application
    cannot locate the original source file (usually because an extension
    has injected text into the sphinx parsing engine).
    """
    source, line = get_source_line(node)

    if source:
        # 'source' can have the form '/some/path:docstring of some.api' but the
        # colons are forbidden on windows, but on posix just passes through.
        if ':docstring of' in source:
            path, *post = source.rpartition(':docstring of')
            post = ''.join(post)
        else:
            path = source
            post = ''
        # We locate references relative to the parent of the doc
        # directory, which for matplotlib, will be the root of the
        # matplotlib repo. When matplotlib is not an editable install
        # weird things will happen, but we can't totally recover from
        # that.
        basepath = Path(app.srcdir).parent.resolve()

        fullpath = Path(path).resolve()

        try:
            path = fullpath.relative_to(basepath)
        except ValueError:
            # Sometimes docs directly contain e.g. docstrings
            # from installed modules, and we record those as
            # <external> so as to be independent of where the
            # module was installed
            path = Path("<external>") / fullpath.name

        # Ensure that all reported paths are POSIX so that docs
        # on windows result in the same warnings in the JSON file.
        path = path.as_posix()

    else:
        path = "<unknown>"
        post = ''
    if not line:
        line = ""

    return f"{path}{post}:{line}"


def _truncate_location(location):
    """
    Cuts off anything after the first colon in location strings.

    This allows for easy comparison even when line numbers change
    (as they do regularly).
    """
    return location.split(":", 1)[0]


def handle_missing_reference(app, domain, node):
    """
    Handle the warn-missing-reference Sphinx event.

    This function will:

    #. record missing references for saving/comparing with ignored list.
    #. prevent Sphinx from raising a warning on ignored references.
    """
    refdomain = node["refdomain"]
    reftype = node["reftype"]
    target = node["reftarget"]
    location = get_location(node, app)
    domain_type = f"{refdomain}:{reftype}"

    app.env.missing_references_events[(domain_type, target)].add(location)

    # If we're ignoring this event, return True so that Sphinx thinks we handled it,
    # even though we didn't print or warn. If we aren't ignoring it, Sphinx will print a
    # warning about the missing reference.
    if location in app.env.missing_references_ignored_references.get(
            (domain_type, target), []):
        return True


def warn_unused_missing_references(app, exc):
    """
    Check that all lines of the existing JSON file are still necessary.
    """
    # We can only warn if we are building from a source install
    # otherwise, we just have to skip this step.
    basepath = Path(matplotlib.__file__).parent.parent.parent.resolve()
    srcpath = Path(app.srcdir).parent.resolve()

    if basepath != srcpath:
        return

    # This is a dictionary of {(domain_type, target): locations}
    references_ignored = app.env.missing_references_ignored_references
    references_events = app.env.missing_references_events

    # Warn about any reference which is no longer missing.
    for (domain_type, target), locations in references_ignored.items():
        missing_reference_locations = [
            _truncate_location(location)
            for location in references_events.get((domain_type, target), [])]

        # For each ignored reference location, ensure a missing reference
        # was observed. If it wasn't observed, issue a warning.
        for ignored_reference_location in locations:
            short_location = _truncate_location(ignored_reference_location)
            if short_location not in missing_reference_locations:
                msg = (f"Reference {domain_type} {target} for "
                       f"{ignored_reference_location} can be removed"
                       f" from {app.config.missing_references_filename}."
                        " It is no longer a missing reference in the docs.")
                logger.warning(msg,
                               location=ignored_reference_location,
                               type='ref',
                               subtype=domain_type)


def save_missing_references(app, exc):
    """
    Write a new JSON file containing missing references.
    """
    json_path = Path(app.confdir) / app.config.missing_references_filename
    references_warnings = app.env.missing_references_events
    _write_missing_references_json(references_warnings, json_path)


def _write_missing_references_json(records, json_path):
    """
    Convert ignored references to a format which we can write as JSON

    Convert from ``{(domain_type, target): locations}`` to
    ``{domain_type: {target: locations}}`` since JSON can't serialize tuples.
    """
    # Sorting records and keys avoids needlessly big diffs when
    # missing_references.json is regenerated.
    transformed_records = defaultdict(dict)
    for (domain_type, target), paths in records.items():
        transformed_records[domain_type][target] = sorted(paths)
    with json_path.open("w") as stream:
        json.dump(transformed_records, stream, sort_keys=True, indent=2)
        stream.write("\n")  # Silence pre-commit no-newline-at-end-of-file warning.


def _read_missing_references_json(json_path):
    """
    Convert from the JSON file to the form used internally by this
    extension.

    The JSON file is stored as ``{domain_type: {target: [locations,]}}``
    since JSON can't store dictionary keys which are tuples. We convert
    this back to ``{(domain_type, target):[locations]}`` for internal use.

    """
    with json_path.open("r") as stream:
        data = json.load(stream)

    ignored_references = {}
    for domain_type, targets in data.items():
        for target, locations in targets.items():
            ignored_references[(domain_type, target)] = locations
    return ignored_references


def prepare_missing_references_setup(app):
    """
    Initialize this extension once the configuration is ready.
    """
    if not app.config.missing_references_enabled:
        # no-op when we are disabled.
        return

    app.connect("warn-missing-reference", handle_missing_reference)
    if app.config.missing_references_warn_unused_ignores:
        app.connect("build-finished", warn_unused_missing_references)
    if app.config.missing_references_write_json:
        app.connect("build-finished", save_missing_references)

    json_path = Path(app.confdir) / app.config.missing_references_filename
    app.env.missing_references_ignored_references = (
        _read_missing_references_json(json_path) if json_path.exists() else {}
    )
    app.env.missing_references_events = defaultdict(set)


def setup(app):
    app.add_config_value("missing_references_enabled", True, "env")
    app.add_config_value("missing_references_write_json", False, "env")
    app.add_config_value("missing_references_warn_unused_ignores", True, "env")
    app.add_config_value("missing_references_filename",
                         "missing-references.json", "env")

    app.connect("builder-inited", prepare_missing_references_setup)

    return {'parallel_read_safe': True}
