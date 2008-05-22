from docutils import nodes
from docutils.writers.html4css1 import HTMLTranslator
from sphinx.latexwriter import LaTeXTranslator

# Define LaTeX math node:
class latex_math(nodes.General, nodes.Element):
    pass

def math_role(role, rawtext, text, lineno, inliner,
              options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    try:
        mathml_tree = parse_latex_math(latex, inline=True)
    except SyntaxError, msg:
        msg = inliner.reporter.error(msg, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['mathml_tree'] = mathml_tree
    return [node], []


try:
    from docutils.parsers.rst import Directive
except ImportError:
    # Register directive the old way:
    from docutils.parsers.rst.directives import _directives
    def math_directive(name, arguments, options, content, lineno,
                       content_offset, block_text, state, state_machine):
        latex = ''.join(content)
        try:
            mathml_tree = parse_latex_math(latex, inline=False)
        except SyntaxError, msg:
            error = state_machine.reporter.error(
                msg, nodes.literal_block(block_text, block_text), line=lineno)
            return [error]
        node = latex_math(block_text)
        node['latex'] = latex
        node['mathml_tree'] = mathml_tree
        return [node]
    math_directive.arguments = None
    math_directive.options = {}
    math_directive.content = 1
    _directives['math'] = math_directive
else:
    class math_directive(Directive):
        has_content = True
        def run(self):
            latex = ' '.join(self.content)
            try:
                mathml_tree = parse_latex_math(latex, inline=False)
            except SyntaxError, msg:
                error = self.state_machine.reporter.error(
                    msg, nodes.literal_block(self.block_text, self.block_text),
                    line=self.lineno)
                return [error]
            node = latex_math(self.block_text)
            node['latex'] = latex
            node['mathml_tree'] = mathml_tree
            return [node]
    from docutils.parsers.rst import directives
    directives.register_directive('math', math_directive)

def setup(app):
    app.add_node(latex_math)
    app.add_role('math', math_role)

    # Add visit/depart methods to HTML-Translator:
    def visit_latex_math_html(self, node):
        mathml = ''.join(node['mathml_tree'].xml())
        self.body.append(mathml)
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


# LaTeX to MathML translation stuff:
class math:
    """Base class for MathML elements."""

    nchildren = 1000000
    """Required number of children"""

    def __init__(self, children=None, inline=None):
        """math([children]) -> MathML element

        children can be one child or a list of children."""

        self.children = []
        if children is not None:
            if type(children) is list:
                for child in children:
                    self.append(child)
            else:
                # Only one child:
                self.append(children)

        if inline is not None:
            self.inline = inline

    def __repr__(self):
        if hasattr(self, 'children'):
            return self.__class__.__name__ + '(%s)' % \
                   ','.join([repr(child) for child in self.children])
        else:
            return self.__class__.__name__

    def full(self):
        """Room for more children?"""

        return len(self.children) >= self.nchildren

    def append(self, child):
        """append(child) -> element

        Appends child and returns self if self is not full or first
        non-full parent."""

        assert not self.full()
        self.children.append(child)
        child.parent = self
        node = self
        while node.full():
            node = node.parent
        return node

    def delete_child(self):
        """delete_child() -> child

        Delete last child and return it."""

        child = self.children[-1]
        del self.children[-1]
        return child

    def close(self):
        """close() -> parent

        Close element and return first non-full element."""

        parent = self.parent
        while parent.full():
            parent = parent.parent
        return parent

    def xml(self):
        """xml() -> xml-string"""

        return self.xml_start() + self.xml_body() + self.xml_end()

    def xml_start(self):
        if not hasattr(self, 'inline'):
            return ['<%s>' % self.__class__.__name__]
        xmlns = 'http://www.w3.org/1998/Math/MathML'
        if self.inline:
            return ['<math xmlns="%s">' % xmlns]
        else:
            return ['<math xmlns="%s" mode="display">' % xmlns]

    def xml_end(self):
        return ['</%s>' % self.__class__.__name__]

    def xml_body(self):
        xml = []
        for child in self.children:
            xml.extend(child.xml())
        return xml

class mrow(math): pass
class mtable(math): pass
class mtr(mrow): pass
class mtd(mrow): pass

class mx(math):
    """Base class for mo, mi, and mn"""

    nchildren = 0
    def __init__(self, data):
        self.data = data

    def xml_body(self):
        return [self.data]

class mo(mx):
    translation = {'<': '&lt;', '>': '&gt;'}
    def xml_body(self):
        return [self.translation.get(self.data, self.data)]

class mi(mx): pass
class mn(mx): pass

class msub(math):
    nchildren = 2

class msup(math):
    nchildren = 2

class msqrt(math):
    nchildren = 1

class mroot(math):
    nchildren = 2

class mfrac(math):
    nchildren = 2

class msubsup(math):
    nchildren = 3
    def __init__(self, children=None, reversed=False):
        self.reversed = reversed
        math.__init__(self, children)

    def xml(self):
        if self.reversed:
##            self.children[1:3] = self.children[2:0:-1]
            self.children[1:3] = [self.children[2], self.children[1]]
            self.reversed = False
        return math.xml(self)

class mfenced(math):
    translation = {'\\{': '{', '\\langle': u'\u2329',
                   '\\}': '}', '\\rangle': u'\u232A',
                   '.': ''}
    def __init__(self, par):
        self.openpar = par
        math.__init__(self)

    def xml_start(self):
        open = self.translation.get(self.openpar, self.openpar)
        close = self.translation.get(self.closepar, self.closepar)
        return ['<mfenced open="%s" close="%s">' % (open, close)]

class mspace(math):
    nchildren = 0

class mstyle(math):
    def __init__(self, children=None, nchildren=None, **kwargs):
        if nchildren is not None:
            self.nchildren = nchildren
        math.__init__(self, children)
        self.attrs = kwargs

    def xml_start(self):
        return ['<mstyle '] + ['%s="%s"' % item
                               for item in self.attrs.items()] + ['>']

class mover(math):
    nchildren = 2
    def __init__(self, children=None, reversed=False):
        self.reversed = reversed
        math.__init__(self, children)

    def xml(self):
        if self.reversed:
            self.children.reverse()
            self.reversed = False
        return math.xml(self)

class munder(math):
    nchildren = 2

class munderover(math):
    nchildren = 3
    def __init__(self, children=None):
        math.__init__(self, children)

class mtext(math):
    nchildren = 0
    def __init__(self, text):
        self.text = text

    def xml_body(self):
        return [self.text]


over = {'tilde': '~',
        'hat': '^',
        'bar': '_',
        'vec': u'\u2192'}

Greek = {
    # Upper case greek letters:
    'Phi': u'\u03a6', 'Xi': u'\u039e', 'Sigma': u'\u03a3', 'Psi': u'\u03a8', 'Delta': u'\u0394', 'Theta': u'\u0398', 'Upsilon': u'\u03d2', 'Pi': u'\u03a0', 'Omega': u'\u03a9', 'Gamma': u'\u0393', 'Lambda': u'\u039b'}
greek = {
    # Lower case greek letters:
    'tau': u'\u03c4', 'phi': u'\u03d5', 'xi': u'\u03be', 'iota': u'\u03b9', 'epsilon': u'\u03f5', 'varrho': u'\u03f1', 'varsigma': u'\u03c2', 'beta': u'\u03b2', 'psi': u'\u03c8', 'rho': u'\u03c1', 'delta': u'\u03b4', 'alpha': u'\u03b1', 'zeta': u'\u03b6', 'omega': u'\u03c9', 'varepsilon': u'\u03b5', 'kappa': u'\u03ba', 'vartheta': u'\u03d1', 'chi': u'\u03c7', 'upsilon': u'\u03c5', 'sigma': u'\u03c3', 'varphi': u'\u03c6', 'varpi': u'\u03d6', 'mu': u'\u03bc', 'eta': u'\u03b7', 'theta': u'\u03b8', 'pi': u'\u03c0', 'varkappa': u'\u03f0', 'nu': u'\u03bd', 'gamma': u'\u03b3', 'lambda': u'\u03bb'}

special = {
    # Binary operation symbols:
    'wedge': u'\u2227', 'diamond': u'\u22c4', 'star': u'\u22c6', 'amalg': u'\u2a3f', 'ast': u'\u2217', 'odot': u'\u2299', 'triangleleft': u'\u25c1', 'bigtriangleup': u'\u25b3', 'ominus': u'\u2296', 'ddagger': u'\u2021', 'wr': u'\u2240', 'otimes': u'\u2297', 'sqcup': u'\u2294', 'oplus': u'\u2295', 'bigcirc': u'\u25cb', 'oslash': u'\u2298', 'sqcap': u'\u2293', 'bullet': u'\u2219', 'cup': u'\u222a', 'cdot': u'\u22c5', 'cap': u'\u2229', 'bigtriangledown': u'\u25bd', 'times': u'\xd7', 'setminus': u'\u2216', 'circ': u'\u2218', 'vee': u'\u2228', 'uplus': u'\u228e', 'mp': u'\u2213', 'dagger': u'\u2020', 'triangleright': u'\u25b7', 'div': u'\xf7', 'pm': u'\xb1',
    # Relation symbols:
    'subset': u'\u2282', 'propto': u'\u221d', 'geq': u'\u2265', 'ge': u'\u2265', 'sqsubset': u'\u228f', 'Join': u'\u2a1d', 'frown': u'\u2322', 'models': u'\u22a7', 'supset': u'\u2283', 'in': u'\u2208', 'doteq': u'\u2250', 'dashv': u'\u22a3', 'gg': u'\u226b', 'leq': u'\u2264', 'succ': u'\u227b', 'vdash': u'\u22a2', 'cong': u'\u2245', 'simeq': u'\u2243', 'subseteq': u'\u2286', 'parallel': u'\u2225', 'equiv': u'\u2261', 'ni': u'\u220b', 'le': u'\u2264', 'approx': u'\u2248', 'precsim': u'\u227e', 'sqsupset': u'\u2290', 'll': u'\u226a', 'sqsupseteq': u'\u2292', 'mid': u'\u2223', 'prec': u'\u227a', 'succsim': u'\u227f', 'bowtie': u'\u22c8', 'perp': u'\u27c2', 'sqsubseteq': u'\u2291', 'asymp': u'\u224d', 'smile': u'\u2323', 'supseteq': u'\u2287', 'sim': u'\u223c', 'neq': u'\u2260',
    # Arrow symbols:
    'searrow': u'\u2198', 'updownarrow': u'\u2195', 'Uparrow': u'\u21d1', 'longleftrightarrow': u'\u27f7', 'Leftarrow': u'\u21d0', 'longmapsto': u'\u27fc', 'Longleftarrow': u'\u27f8', 'nearrow': u'\u2197', 'hookleftarrow': u'\u21a9', 'downarrow': u'\u2193', 'Leftrightarrow': u'\u21d4', 'longrightarrow': u'\u27f6', 'rightharpoondown': u'\u21c1', 'longleftarrow': u'\u27f5', 'rightarrow': u'\u2192', 'Updownarrow': u'\u21d5', 'rightharpoonup': u'\u21c0', 'Longleftrightarrow': u'\u27fa', 'leftarrow': u'\u2190', 'mapsto': u'\u21a6', 'nwarrow': u'\u2196', 'uparrow': u'\u2191', 'leftharpoonup': u'\u21bc', 'leftharpoondown': u'\u21bd', 'Downarrow': u'\u21d3', 'leftrightarrow': u'\u2194', 'Longrightarrow': u'\u27f9', 'swarrow': u'\u2199', 'hookrightarrow': u'\u21aa', 'Rightarrow': u'\u21d2',
    # Miscellaneous symbols:
    'infty': u'\u221e', 'surd': u'\u221a', 'partial': u'\u2202', 'ddots': u'\u22f1', 'exists': u'\u2203', 'flat': u'\u266d', 'diamondsuit': u'\u2662', 'wp': u'\u2118', 'spadesuit': u'\u2660', 'Re': u'\u211c', 'vdots': u'\u22ee', 'aleph': u'\u2135', 'clubsuit': u'\u2663', 'sharp': u'\u266f', 'angle': u'\u2220', 'prime': u'\u2032', 'natural': u'\u266e', 'ell': u'\u2113', 'neg': u'\xac', 'top': u'\u22a4', 'nabla': u'\u2207', 'bot': u'\u22a5', 'heartsuit': u'\u2661', 'cdots': u'\u22ef', 'Im': u'\u2111', 'forall': u'\u2200', 'imath': u'\u0131', 'hbar': u'\u210f', 'emptyset': u'\u2205',
    # Variable-sized symbols:
    'bigotimes': u'\u2a02', 'coprod': u'\u2210', 'int': u'\u222b', 'sum': u'\u2211', 'bigodot': u'\u2a00', 'bigcup': u'\u22c3', 'biguplus': u'\u2a04', 'bigcap': u'\u22c2', 'bigoplus': u'\u2a01', 'oint': u'\u222e', 'bigvee': u'\u22c1', 'bigwedge': u'\u22c0', 'prod': u'\u220f',
    # Braces:
    'langle': u'\u2329', 'rangle': u'\u232A'}

sumintprod = ''.join([special[symbol] for symbol in
                      ['sum', 'int', 'oint', 'prod']])

functions = ['arccos', 'arcsin', 'arctan', 'arg', 'cos',  'cosh',
             'cot',    'coth',   'csc',    'deg', 'det',  'dim',
             'exp',    'gcd',    'hom',    'inf', 'ker',  'lg',
             'lim',    'liminf', 'limsup', 'ln',  'log',  'max',
             'min',    'Pr',     'sec',    'sin', 'sinh', 'sup',
             'tan',    'tanh',
             'injlim',  'varinjlim', 'varlimsup',
             'projlim', 'varliminf', 'varprojlim']


def parse_latex_math(string, inline=True):
    """parse_latex_math(string [,inline]) -> MathML-tree

    Returns a MathML-tree parsed from string.  inline=True is for
    inline math and inline=False is for displayed math.

    tree is the whole tree and node is the current element."""

    # Normalize white-space:
    string = ' '.join(string.split())

    if inline:
        node = mrow()
        tree = math(node, inline=True)
    else:
        node = mtd()
        tree = math(mtable(mtr(node)), inline=False)

    while len(string) > 0:
        n = len(string)
        c = string[0]
        skip = 1  # number of characters consumed
        if n > 1:
            c2 = string[1]
        else:
            c2 = ''
##        print n, string, c, c2, node.__class__.__name__
        if c == ' ':
            pass
        elif c == '\\':
            if c2 in '{}':
                node = node.append(mo(c2))
                skip = 2
            elif c2 == ' ':
                node = node.append(mspace())
                skip = 2
            elif c2.isalpha():
                # We have a LaTeX-name:
                i = 2
                while i < n and string[i].isalpha():
                    i += 1
                name = string[1:i]
                node, skip = handle_keyword(name, node, string[i:])
                skip += i
            elif c2 == '\\':
                # End of a row:
                entry = mtd()
                row = mtr(entry)
                node.close().close().append(row)
                node = entry
                skip = 2
            else:
                raise SyntaxError('Syntax error: "%s%s"' % (c, c2))
        elif c.isalpha():
            node = node.append(mi(c))
        elif c.isdigit():
            node = node.append(mn(c))
        elif c in "+-/()[]|=<>,.!'":
            node = node.append(mo(c))
        elif c == '_':
            child = node.delete_child()
            if isinstance(child, msup):
                sub = msubsup(child.children, reversed=True)
            elif isinstance(child, mo) and child.data in sumintprod:
                sub = munder(child)
            else:
                sub = msub(child)
            node.append(sub)
            node = sub
        elif c == '^':
            child = node.delete_child()
            if isinstance(child, msub):
                sup = msubsup(child.children)
            elif isinstance(child, mo) and child.data in sumintprod:
                sup = mover(child)
            elif (isinstance(child, munder) and
                  child.children[0].data in sumintprod):
                sup = munderover(child.children)
            else:
                sup = msup(child)
            node.append(sup)
            node = sup
        elif c == '{':
            row = mrow()
            node.append(row)
            node = row
        elif c == '}':
            node = node.close()
        elif c == '&':
            entry = mtd()
            node.close().append(entry)
            node = entry
        else:
            raise SyntaxError('Illegal character: "%s"' % c)
        string = string[skip:]
    return tree


mathbb = {'A': u'\U0001D538',
          'B': u'\U0001D539',
          'C': u'\u2102',
          'D': u'\U0001D53B',
          'E': u'\U0001D53C',
          'F': u'\U0001D53D',
          'G': u'\U0001D53E',
          'H': u'\u210D',
          'I': u'\U0001D540',
          'J': u'\U0001D541',
          'K': u'\U0001D542',
          'L': u'\U0001D543',
          'M': u'\U0001D544',
          'N': u'\u2115',
          'O': u'\U0001D546',
          'P': u'\u2119',
          'Q': u'\u211A',
          'R': u'\u211D',
          'S': u'\U0001D54A',
          'T': u'\U0001D54B',
          'U': u'\U0001D54C',
          'V': u'\U0001D54D',
          'W': u'\U0001D54E',
          'X': u'\U0001D54F',
          'Y': u'\U0001D550',
          'Z': u'\u2124'}

negatables = {'=': u'\u2260',
              '\in': u'\u2209',
              '\equiv': u'\u2262'}


def handle_keyword(name, node, string):
    skip = 0
    if len(string) > 0 and string[0] == ' ':
        string = string[1:]
        skip = 1
    if name == 'begin':
        if not string.startswith('{matrix}'):
            raise SyntaxError('Expected "\begin{matrix}"!')
        skip += 8
        entry = mtd()
        table = mtable(mtr(entry))
        node.append(table)
        node = entry
    elif name == 'end':
        if not string.startswith('{matrix}'):
            raise SyntaxError('Expected "\end{matrix}"!')
        skip += 8
        node = node.close().close().close()
    elif name == 'text':
        if string[0] != '{':
            raise SyntaxError('Expected "\text{...}"!')
        i = string.find('}')
        if i == -1:
            raise SyntaxError('Expected "\text{...}"!')
        node = node.append(mtext(string[1:i]))
        skip += i + 1
    elif name == 'sqrt':
        sqrt = msqrt()
        node.append(sqrt)
        node = sqrt
    elif name == 'frac':
        frac = mfrac()
        node.append(frac)
        node = frac
    elif name == 'left':
        for par in ['(', '[', '|', '\\{', '\\langle', '.']:
            if string.startswith(par):
                break
        else:
            raise SyntaxError('Missing left-brace!')
        fenced = mfenced(par)
        node.append(fenced)
        row = mrow()
        fenced.append(row)
        node = row
        skip += len(par)
    elif name == 'right':
        for par in [')', ']', '|', '\\}', '\\rangle', '.']:
            if string.startswith(par):
                break
        else:
            raise SyntaxError('Missing right-brace!')
        node = node.close()
        node.closepar = par
        node = node.close()
        skip += len(par)
    elif name == 'not':
        for operator in negatables:
            if string.startswith(operator):
                break
        else:
            raise SyntaxError('Expected something to negate: "\\not ..."!')
        node = node.append(mo(negatables[operator]))
        skip += len(operator)
    elif name == 'mathbf':
        style = mstyle(nchildren=1, fontweight='bold')
        node.append(style)
        node = style
    elif name == 'mathbb':
        if string[0] != '{' or not string[1].isupper() or string[2] != '}':
            raise SyntaxError('Expected something like "\mathbb{A}"!')
        node = node.append(mi(mathbb[string[1]]))
        skip += 3
    elif name in greek:
        node = node.append(mi(greek[name]))
    elif name in Greek:
        node = node.append(mo(Greek[name]))
    elif name in special:
        node = node.append(mo(special[name]))
    elif name in functions:
        node = node.append(mo(name))
    else:
        chr = over.get(name)
        if chr is not None:
            ovr = mover(mo(chr), reversed=True)
            node.append(ovr)
            node = ovr
        else:
            raise SyntaxError('Unknown LaTeX command: ' + name)

    return node, skip
