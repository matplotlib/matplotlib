import os
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.writers.html4css1 import HTMLTranslator
from sphinx.latexwriter import LaTeXTranslator
import warnings

# Define LaTeX math node:
class latex_math(nodes.General, nodes.Element):
    pass

def fontset_choice(arg):
    return directives.choice(arg, ['cm', 'stix', 'stixsans'])

options_spec = {'fontset': fontset_choice}

def math_role(role, rawtext, text, lineno, inliner,
              options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    return [node], []
math_role.options = options_spec

def math_directive_run(content, block_text, options):
    latex = ''.join(content)
    node = latex_math(block_text)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    return [node]

try:
    from docutils.parsers.rst import Directive
except ImportError:
    # Register directive the old way:
    from docutils.parsers.rst.directives import _directives
    def math_directive(name, arguments, options, content, lineno,
                       content_offset, block_text, state, state_machine):
        return math_directive_run(content, block_text, options)
    math_directive.arguments = None
    math_directive.options = options_spec
    math_directive.content = 1
    _directives['math'] = math_directive
else:
    class math_directive(Directive):
        has_content = True
        option_spec = options_spec

        def run(self):
            return math_directive_run(self.content, self.block_text,
                                      self.options)
    from docutils.parsers.rst import directives
    directives.register_directive('math', math_directive)

def setup(app):
    app.add_node(latex_math)
    app.add_role('math', math_role)

    # Add visit/depart methods to HTML-Translator:
    def visit_latex_math_html(self, node):
        source = self.document.attributes['source']
        self.body.append(latex2html(node, source))
    def depart_latex_math_html(self, node):
            pass
    HTMLTranslator.visit_latex_math = visit_latex_math_html
    HTMLTranslator.depart_latex_math = depart_latex_math_html

    # Add visit/depart methods to LaTeX-Translator:
    def visit_latex_math_latex(self, node):
        inline = isinstance(node.parent, nodes.TextElement)
        if inline:
            self.body.append('$%s$' % node['latex'])
        else:
            self.body.extend(['\\begin{equation}',
                              node['latex'],
                              '\\end{equation}'])
    def depart_latex_math_latex(self, node):
            pass
    LaTeXTranslator.visit_latex_math = visit_latex_math_latex
    LaTeXTranslator.depart_latex_math = depart_latex_math_latex

from matplotlib import rcParams
from matplotlib.mathtext import MathTextParser
rcParams['mathtext.fontset'] = 'cm'
mathtext_parser = MathTextParser("Bitmap")


# This uses mathtext to render the expression
def latex2png(latex, filename, fontset='cm'):
    latex = "$%s$" % latex
    orig_fontset = rcParams['mathtext.fontset']
    rcParams['mathtext.fontset'] = fontset
    if os.path.exists(filename):
        depth = mathtext_parser.get_depth(latex, dpi=100)
    else:
        try:
            depth = mathtext_parser.to_png(filename, latex, dpi=100)
        except:
            warnings.warn("Could not render math expression %s" % latex,
                          Warning)
            depth = 0
    rcParams['mathtext.fontset'] = orig_fontset
    return depth

# LaTeX to HTML translation stuff:
def latex2html(node, source):
    inline = isinstance(node.parent, nodes.TextElement)
    latex = node['latex']
    name = 'math-%s' % md5(latex).hexdigest()[-10:]
    dest = '_static/%s.png' % name
    depth = latex2png(latex, dest, node['fontset'])

    path = '_static'
    count = source.split('/doc/')[-1].count('/')
    for i in range(count):
        if os.path.exists(path): break
        path = '../'+path
    path = '../'+path #specifically added for matplotlib
    if inline:
        cls = ''
    else:
        cls = 'class="center" '
    if inline and depth != 0:
        style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
    else:
        style = ''

    return '<img src="%s/%s.png" %s%s/>' % (path, name, cls, style)

