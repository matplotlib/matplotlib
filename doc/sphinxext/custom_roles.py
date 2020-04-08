from docutils import nodes
from os.path import sep
from matplotlib import rcParamsDefault


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    rendered = nodes.Text(f'rcParams["{text}"]')

    source = inliner.document.attributes['source'].replace(sep, '/')
    rel_source = source.split('/doc/', 1)[1]

    levels = rel_source.count('/')
    refuri = ('../' * levels +
              'tutorials/introductory/customizing.html' +
              f"?highlight={text}#a-sample-matplotlibrc-file")

    ref = nodes.reference(rawtext, rendered, refuri=refuri)
    node_list = [nodes.literal('', '', ref)]
    if text in rcParamsDefault:
        node_list.extend([
            nodes.Text(' (default: '),
            nodes.literal('', repr(rcParamsDefault[text])),
            nodes.Text(')'),
            ])
    return node_list, []


def setup(app):
    app.add_role("rc", rcparam_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
