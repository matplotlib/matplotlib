from docutils import nodes


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    rendered = nodes.Text('rcParams["{}"]'.format(text))
    return [nodes.literal(rawtext, rendered)], []


def setup(app):
    app.add_role("rc", rcparam_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
