# Skip deprecated members


def skip_deprecated(app, what, name, obj, skip, options):
    if skip:
        return skip
    skipped = {"matplotlib.colors": ["ColorConverter", "hex2color", "rgb2hex"]}
    skip_list = skipped.get(getattr(obj, "__module__", None))
    if skip_list is not None:
        return getattr(obj, "__name__", None) in skip_list


def setup(app):
    app.connect('autodoc-skip-member', skip_deprecated)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
