from __future__ import print_function
symbols = [
    ["Lower-case Greek",
     5,
     r"""\alpha \beta \gamma \chi \delta \epsilon \eta \iota \kappa
         \lambda \mu \nu \omega \phi \pi \psi \rho \sigma \tau \theta
         \upsilon \xi \zeta \digamma \varepsilon \varkappa \varphi
         \varpi \varrho \varsigma \vartheta"""],
    ["Upper-case Greek",
     6,
     r"""\Delta \Gamma \Lambda \Omega \Phi \Pi \Psi \Sigma \Theta
     \Upsilon \Xi \mho \nabla"""],
    ["Hebrew",
     4,
     r"""\aleph \beth \daleth \gimel"""],
    ["Delimiters",
     6,
     r"""| \{ \lfloor / \Uparrow \llcorner \vert \} \rfloor \backslash
         \uparrow \lrcorner \| \langle \lceil [ \Downarrow \ulcorner
         \Vert \rangle \rceil ] \downarrow \urcorner"""],
    ["Big symbols",
     5,
     r"""\bigcap \bigcup \bigodot \bigoplus \bigotimes \biguplus
         \bigvee \bigwedge \coprod \oint \prod \sum \int"""],
    ["Standard function names",
     4,
     r"""\arccos \csc \ker \min \arcsin \deg \lg \Pr \arctan \det \lim
         \gcd \ln \sup \cot \hom \log \tan \coth \inf \max \tanh
         \sec \arg \dim \liminf \sin \cos \exp \limsup \sinh \cosh"""],
    ["Binary operation and relation symbols",
     3,
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
     """],
    ["Arrow symbols",
     2,
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
     """],
    ["Miscellaneous symbols",
     3,
     r"""\neg \infty \forall \wp \exists \bigstar \angle \partial
     \nexists \measuredangle \eth \emptyset \sphericalangle \clubsuit
     \varnothing \complement \diamondsuit \imath \Finv \triangledown
     \heartsuit \jmath \Game \spadesuit \ell \hbar \vartriangle \cdots
     \hslash \vdots \blacksquare \ldots \blacktriangle \ddots \sharp
     \prime \blacktriangledown \Im \flat \backprime \Re \natural
     \circledS \P \copyright \ss \circledR \S \yen \AA \checkmark \$
     \iiint \iint \iint \oiiint"""]
]

def run(state_machine):
    def get_n(n, l):
        part = []
        for x in l:
            part.append(x)
            if len(part) == n:
                yield part
                part = []
        yield part

    lines = []
    for category, columns, syms in symbols:
        syms = syms.split()
        syms.sort()
        lines.append("**%s**" % category)
        lines.append('')
        max_width = 0
        for sym in syms:
            max_width = max(max_width, len(sym))
        max_width = max_width * 2 + 16
        header = "    " + (('=' * max_width) + ' ') * columns
        format = '%%%ds' % max_width
        for chunk in get_n(20, get_n(columns, syms)):
            lines.append(header)
            for part in chunk:
                line = []
                for sym in part:
                    line.append(format % (":math:`%s` ``%s``" % (sym, sym)))
                lines.append("    " + " ".join(line))
            lines.append(header)
            lines.append('')

    state_machine.insert_input(lines, "Symbol table")
    return []

def math_symbol_table_directive(name, arguments, options, content, lineno,
                                content_offset, block_text, state, state_machine):
    return run(state_machine)

def setup(app):
    app.add_directive(
        'math_symbol_table', math_symbol_table_directive,
        False, (0, 1, 0))

if __name__ == "__main__":
    # Do some verification of the tables
    from matplotlib import _mathtext_data

    print("SYMBOLS NOT IN STIX:")
    all_symbols = {}
    for category, columns, syms in symbols:
        if category == "Standard Function Names":
            continue
        syms = syms.split()
        for sym in syms:
            if len(sym) > 1:
                all_symbols[sym[1:]] = None
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)

    print("SYMBOLS NOT IN TABLE:")
    for sym in _mathtext_data.tex2uni:
        if sym not in all_symbols:
            print(sym)
