from docutils import nodes
from os.path import sep
from matplotlib import rcParamsDefault


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    try:
        default_str = f' = {rcParamsDefault[text]!r}'
    except KeyError:
        # handling of generic references such as rcParams["figure.subplot.*"]
        default_str = ''
    rendered = nodes.Text(f'rcParams["{text}"]' + default_str)

    source = inliner.document.attributes['source'].replace(sep, '/')
    rel_source = source.split('/doc/', 1)[1]

    levels = rel_source.count('/')
    refuri = ('../' * levels +
              'tutorials/introductory/customizing.html' +
              f"?highlight={text}#a-sample-matplotlibrc-file")

    ref = nodes.reference(rawtext, rendered, refuri=refuri)
    return [nodes.literal('', '', ref)], []


def setup(app):
    app.add_role("rc", rcparam_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
