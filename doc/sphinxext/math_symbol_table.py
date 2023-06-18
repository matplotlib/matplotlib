from docutils.parsers.rst import Directive

from matplotlib import _mathtext, _mathtext_data


symbols = [
    ["Lower-case Greek",
     6,
     r"""\alpha \beta \gamma \delta \epsilon \zeta \eta \theta \iota \kappa \lambda \mu
         \nu \xi \pi \rho \sigma \tau \upsilon \phi  \psi  \chi \omega \digamma
         \varepsilon \vartheta \varkappa \varphi \varpi \varrho \varsigma""".split()],
    ["Upper-case Greek",
     6,
     r"""\Gamma \Delta  \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega
      """.split()],
    ["Hebrew",
     6,
     r"""\aleph \beth \gimel \daleth""".split()],
    ["Delimiters",
     6,
     _mathtext.Parser._delims],
    ["Big symbols",
     6,
     _mathtext.Parser._overunder_symbols | _mathtext.Parser._dropsub_symbols],
    ["Standard function names",
     6,
     {fr"\{fn}" for fn in _mathtext.Parser._function_names}],
    ["Binary operation and relation symbols",
     4,
     set(r"""\ast \pm \slash \cap \star \mp \cup \cdot \uplus
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
     """.split())],
    ["Arrow symbols",
     4,
     set(r"""\leftarrow \longleftarrow \uparrow \Leftarrow \Longleftarrow
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
     """.split())],
    ["Miscellaneous symbols",
     4,
     set(r"""\neg \infty \forall \wp \exists \bigstar \angle \partial
     \nexists \measuredangle \eth \emptyset \sphericalangle \clubsuit
     \varnothing \complement \diamondsuit \imath \Finv \triangledown
     \heartsuit \jmath \Game \spadesuit \ell \hbar \vartriangle \cdots
     \hslash \vdots \blacksquare \ldots \blacktriangle \ddots \sharp
     \prime \blacktriangledown \Im \flat \backprime \Re \natural
     \circledS \P \copyright \ss \circledR \S \yen \AA \checkmark \$
     \cent \triangle \QED \sinewave \mho \nabla""".split())]
]


def run(state_machine):
    def render_symbol(sym):
        if sym.startswith("\\"):
            sym = sym[1:]
            if sym not in (_mathtext.Parser._overunder_functions |
                           _mathtext.Parser._function_names):
                sym = chr(_mathtext_data.tex2uni[sym])
        return f'\\{sym}' if sym in ('\\', '|') else sym

    lines = []
    for category, columns, syms in symbols:
        # Assume that lists are correctly sorted
        if isinstance(syms, set):
            syms = sorted(list(syms))
        columns = min(columns, len(syms))
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
