from docutils import nodes
from os.path import sep
from matplotlib import rcParamsDefault

def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    
    param = rcParamsDefault.get(text)
    if isinstance(param, str):
        txt = f' : "{param}"'
    else:
        txt = f' : {param}'
    rendered = nodes.Text(f'rcParams["{text}"]' + txt)

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
