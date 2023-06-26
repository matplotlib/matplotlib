from docutils.parsers.rst import Directive

from matplotlib import _mathtext, _mathtext_data


symbols = [
    ["Lower-case Greek",
     (r"\alpha", r"\beta", r"\gamma",  r"\chi", r"\delta", r"\epsilon",
      r"\eta", r"\iota",  r"\kappa", r"\lambda", r"\mu", r"\nu",  r"\omega",
      r"\phi",  r"\pi", r"\psi", r"\rho",  r"\sigma",  r"\tau", r"\theta",
      r"\upsilon", r"\xi", r"\zeta",  r"\digamma", r"\varepsilon", r"\varkappa",
      r"\varphi", r"\varpi", r"\varrho", r"\varsigma",  r"\vartheta")],
    ["Upper-case Greek",
     (r"\Delta", r"\Gamma", r"\Lambda", r"\Omega", r"\Phi", r"\Pi", r"\Psi",
      r"\Sigma", r"\Theta", r"\Upsilon", r"\Xi")],
    ["Hebrew",
     (r"\aleph", r"\beth", r"\gimel", r"\daleth")],
    ["Delimiters",
     _mathtext.Parser._delims],
    ["Big symbols",
     _mathtext.Parser._overunder_symbols | _mathtext.Parser._dropsub_symbols],
    ["Standard function names",
     {fr"\{fn}" for fn in _mathtext.Parser._function_names}],
    ["Binary operation and relation symbols",
     r"""\ast \pm \slash \cap \star \mp \cup \cdot \uplus
     \triangleleft \circ \odot \sqcap \triangleright \bullet \ominus
     \sqcup \bigcirc \oplus \wedge \diamond \oslash \vee
     \bigtriangledown \times \otimes \dag \bigtriangleup \div \wr
     \ddag \barwedge \veebar \boxplus \curlywedge \curlyvee \boxminus
     \Cap \Cup \boxtimes \bot \top \dotplus \boxdot \intercal
     \rightthreetimes \divideontimes \leftthreetimes \equiv \leq \geq
     \perp \cong \prec \succ \mid \neq \preceq \succeq \parallel \sim
     \ll \gg \bowtie \simeq \subset \supset \Join \approx \subseteq
     \supseteq \ltimes \asymp \sqsubset \sqsupset \rtimes \doteq
     \sqsubseteq \sqsupseteq \smile \propto \dashv \vdash \frown
     \models \in \ni \notin \approxeq \leqq \geqq \lessgtr \leqslant
     \geqslant \lesseqgtr \backsim \lessapprox \gtrapprox \lesseqqgtr
     \backsimeq \lll \ggg \gtreqqless \triangleq \lessdot \gtrdot
     \gtreqless \circeq \lesssim \gtrsim \gtrless \bumpeq \eqslantless
     \eqslantgtr \backepsilon \Bumpeq \precsim \succsim \between
     \doteqdot \precapprox \succapprox \pitchfork \Subset \Supset
     \fallingdotseq \subseteqq \supseteqq \risingdotseq \sqsubset
     \sqsupset \varpropto \preccurlyeq \succcurlyeq \Vdash \therefore
     \curlyeqprec \curlyeqsucc \vDash \because \blacktriangleleft
     \blacktriangleright \Vvdash \eqcirc \trianglelefteq
     \trianglerighteq \neq \vartriangleleft \vartriangleright \ncong
     \nleq \ngeq \nsubseteq \nmid \nsupseteq \nparallel \nless \ngtr
     \nprec \nsucc \subsetneq \nsim \supsetneq \nVDash \precnapprox
     \succnapprox \subsetneqq \nvDash \precnsim \succnsim \supsetneqq
     \nvdash \lnapprox \gnapprox \ntriangleleft \ntrianglelefteq
     \lneqq \gneqq \ntriangleright \lnsim \gnsim \ntrianglerighteq
     \coloneq \eqsim \nequiv \napprox \nsupset \doublebarwedge \nVdash
     \Doteq \nsubset \eqcolon \ne
     """.split()],
    ["Arrow symbols",
     r"""\leftarrow \longleftarrow \uparrow \Leftarrow \Longleftarrow
     \Uparrow \rightarrow \longrightarrow \downarrow \Rightarrow
     \Longrightarrow \Downarrow \leftrightarrow \updownarrow
     \longleftrightarrow \updownarrow \Leftrightarrow
     \Longleftrightarrow \Updownarrow \mapsto \longmapsto \nearrow
     \hookleftarrow \hookrightarrow \searrow \leftharpoonup
     \rightharpoonup \swarrow \leftharpoondown \rightharpoondown
     \nwarrow \rightleftharpoons \leadsto \dashrightarrow
     \dashleftarrow \leftleftarrows \leftrightarrows \Lleftarrow
     \Rrightarrow \twoheadleftarrow \leftarrowtail \looparrowleft
     \leftrightharpoons \curvearrowleft \circlearrowleft \Lsh
     \upuparrows \upharpoonleft \downharpoonleft \multimap
     \leftrightsquigarrow \rightrightarrows \rightleftarrows
     \rightrightarrows \rightleftarrows \twoheadrightarrow
     \rightarrowtail \looparrowright \rightleftharpoons
     \curvearrowright \circlearrowright \Rsh \downdownarrows
     \upharpoonright \downharpoonright \rightsquigarrow \nleftarrow
     \nrightarrow \nLeftarrow \nRightarrow \nleftrightarrow
     \nLeftrightarrow \to \Swarrow \Searrow \Nwarrow \Nearrow
     \leftsquigarrow
     """.split()],
    ["Miscellaneous symbols",
     r"""\neg \infty \forall \wp \exists \bigstar \angle \partial
     \nexists \measuredangle \eth \emptyset \sphericalangle \clubsuit
     \varnothing \complement \diamondsuit \imath \Finv \triangledown
     \heartsuit \jmath \Game \spadesuit \ell \hbar \vartriangle \cdots
     \hslash \vdots \blacksquare \ldots \blacktriangle \ddots \sharp
     \prime \blacktriangledown \Im \flat \backprime \Re \natural
     \circledS \P \copyright \ss \circledR \S \yen \AA \checkmark \$
     \cent \triangle \QED \sinewave \nabla \mho""".split()]
]


def run(state_machine):

    def render_symbol(sym, ignore_variant=False):
        if ignore_variant and sym != r"\varnothing":
            sym = sym.replace(r"\var", "\\")
        if sym.startswith("\\"):
            sym = sym.lstrip("\\")
            if sym not in (_mathtext.Parser._overunder_functions |
                           _mathtext.Parser._function_names):
                sym = chr(_mathtext_data.tex2uni[sym])
        return f'\\{sym}' if sym in ('\\', '|') else sym

    def columns_calculation(my_list):
        remainder = max_columns = columns = 10
        max_remainder = 0
        for columns_number in range(max_columns - 1, 3, -1):
            remainder = len(my_list) % columns_number
            if remainder > max_remainder:
                columns = columns_number

        return columns

    lines = []
    for category, syms in symbols:
        syms = sorted(syms,
                      # Sort by Unicode and place variants immediately
                      # after standard versions.
                      key=lambda sym: (render_symbol(sym, ignore_variant=True),
                                       sym.startswith(r"\var")),
                      reverse=(category == "Hebrew"))  # Hebrew is rtl
        columns = columns_calculation(syms)
        lines.append("**%s**" % category)
        lines.append('')
        max_width = max(map(len, syms)) * 2 + 16
        header = "    " + (('=' * max_width) + ' ') * columns
        lines.append(header)
        for part in range(0, len(syms), columns):
            row = " ".join(
                f"{render_symbol(sym)} ``{sym}``".rjust(max_width)
                for sym in syms[part:part + columns])
            lines.append(f"    {row}")
        lines.append(header)
        lines.append('')

    state_machine.insert_input(lines, "Symbol table")
    return []


class MathSymbolTableDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        return run(self.state_machine)


def setup(app):
    app.add_directive("math_symbol_table", MathSymbolTableDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata


if __name__ == "__main__":
    # Do some verification of the tables

    print("SYMBOLS NOT IN STIX:")
    all_symbols = {}
    for category, columns, syms in symbols:
        if category == "Standard Function Names":
            continue
        for sym in syms:
            if len(sym) > 1:
                all_symbols[sym[1:]] = None
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)

    print("SYMBOLS NOT IN TABLE:")
    for sym in _mathtext_data.tex2uni:
        if sym not in all_symbols:
            print(sym)
