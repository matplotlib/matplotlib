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

import json
import os.path


def record_missing_reference_handler(app, env, node, contnode):
    if not app.config.missing_references_write_json:
        # no-op when we are disabled.
        return

    if not hasattr(env, "missing_reference_record"):
        env.missing_reference_record = []
    record = env.missing_reference_record

    domain = node["refdomain"]
    typ = node["reftype"]
    target = node["reftarget"]

    dtype = "{}:{}".format(domain, typ)

    record.append([dtype, target])


def save_missing_references_handler(app, exc):
    if not app.config.missing_references_write_json:
        # no-op when we are disabled.
        return

    path = os.path.join(app.confdir, "missing-references.json")
    with open(path, "w") as stream:
        json.dump(app.env.missing_reference_record, stream)


def prepare_missing_references_handler(app, config):
    if config.missing_references_write_json:
        return

    path = os.path.join(app.confdir, "missing-references.json")
    if not os.path.exists(path):
        return

    with open(path, "r") as stream:
        nitpick_ignore = json.load(stream)

    config.nitpick_ignore.extend(tuple(item) for item in nitpick_ignore)


def setup(app):
    app.add_config_value("missing_references_write_json", False, "env")

    app.connect("config-inited", prepare_missing_references_handler)
    app.connect("missing-reference", record_missing_reference_handler)
    app.connect("build-finished", save_missing_references_handler)
