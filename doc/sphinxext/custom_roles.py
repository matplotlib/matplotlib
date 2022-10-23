from urllib.parse import urlsplit, urlunsplit

from docutils import nodes

from matplotlib import rcParamsDefault


class QueryReference(nodes.Inline, nodes.TextElement):
    """
    Wraps a reference or pending reference to add a query string.

    The query string is generated from the attributes added to this node.

    Also equivalent to a `~docutils.nodes.literal` node.
    """

    def to_query_string(self):
        """Generate query string from node attributes."""
        return '&'.join(f'{name}={value}' for name, value in self.attlist())


def visit_query_reference_node(self, node):
    """
    Resolve *node* into query strings on its ``reference`` children.

    Then act as if this is a `~docutils.nodes.literal`.
    """
    query = node.to_query_string()
    for refnode in node.findall(nodes.reference):
        uri = urlsplit(refnode['refuri'])._replace(query=query)
        refnode['refuri'] = urlunsplit(uri)

    self.visit_literal(node)


def depart_query_reference_node(self, node):
    """
    Act as if this is a `~docutils.nodes.literal`.
    """
    self.depart_literal(node)


def rcparam_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Generate a pending cross-reference so that Sphinx will ensure this link
    # isn't broken at some point in the future.
    title = f'rcParams["{text}"]'
    target = 'matplotlibrc-sample'
    ref_nodes, messages = inliner.interpreted(title, f'{title} <{target}>',
                                              'ref', lineno)

    qr = QueryReference(rawtext, highlight=text)
    qr += ref_nodes
    node_list = [qr]

    # The default backend would be printed as "agg", but that's not correct (as
    # the default is actually determined by fallback).
    if text in rcParamsDefault and text != "backend":
        node_list.extend([
            nodes.Text(' (default: '),
            nodes.literal('', repr(rcParamsDefault[text])),
            nodes.Text(')'),
            ])

    return node_list, messages


def setup(app):
    app.add_role("rc", rcparam_role)
    app.add_node(
        QueryReference,
        html=(visit_query_reference_node, depart_query_reference_node),
        latex=(visit_query_reference_node, depart_query_reference_node),
        text=(visit_query_reference_node, depart_query_reference_node),
    )
    return {"parallel_read_safe": True, "parallel_write_safe": True}
