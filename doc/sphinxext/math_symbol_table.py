import re
from docutils.parsers.rst import Directive

from matplotlib import _mathtext, _mathtext_data

bb_pattern = re.compile("Bbb[A-Z]")
scr_pattern = re.compile("scr[a-zA-Z]")
frak_pattern = re.compile("frak[A-Z]")

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
    ["Latin named characters",
     r"""\aa \AA \ae \AE \oe \OE \O \o \thorn \Thorn \ss \eth \dh \DH""".split()],
    ["Delimiters",
     _mathtext.Parser._delims],
    ["Big symbols",
     _mathtext.Parser._overunder_symbols | _mathtext.Parser._dropsub_symbols],
    ["Standard function names",
     {fr"\{fn}" for fn in _mathtext.Parser._function_names}],
    ["Binary operation symbols",
     _mathtext.Parser._binary_operators],
    ["Relation symbols",
     _mathtext.Parser._relation_symbols],
    ["Arrow symbols",
     _mathtext.Parser._arrow_symbols],
    ["Dot symbols",
     r"""\cdots \vdots \ldots \ddots \adots \Colon \therefore \because""".split()],
    ["Black-board characters",
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(bb_pattern, symbol)]],
    ["Script characters",
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(scr_pattern, symbol)]],
    ["Fraktur characters",
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(frak_pattern, symbol)]],
    ["Miscellaneous symbols",
     r"""\neg \infty \forall \wp \exists \bigstar \angle \partial
     \nexists \measuredangle \emptyset \sphericalangle \clubsuit
     \varnothing \complement \diamondsuit \imath \Finv \triangledown
     \heartsuit \jmath \Game \spadesuit \ell \hbar \vartriangle
     \hslash \blacksquare \blacktriangle \sharp \increment
     \prime \blacktriangledown \Im \flat \backprime \Re \natural
     \circledS \P \copyright \circledR \S \yen \checkmark \$
     \cent \triangle \QED \sinewave \dag \ddag \perthousand \ac
     \lambdabar \L \l \degree \danger \maltese \clubsuitopen
     \i \hermitmatrix \sterling \nabla \mho""".split()],
]


def calculate_best_columns(
        symbols_list, max_line_length, max_columns
):
    """
    Calculate the best number of columns to fit the symbols within the
    given constraints.

    Parameters
    ----------
    symbols_list : list of str
        List symbols to be displayed.
    max_line_length : int
        Maximum allowed length of a line.
    max_columns : int
        Maximum number of columns to consider.

    Returns
    -------
    int
        Number of columns that fits within the constraints.
    """
    max_cell_width = max(len(sym) for sym in symbols_list)
    return min(max_columns, len(symbols_list), max_line_length // max_cell_width)


def run(state_machine):

    def render_symbol(sym, ignore_variant=False):
        if ignore_variant and sym not in (r"\varnothing", r"\varlrtriangle"):
            sym = sym.replace(r"\var", "\\")
        if sym.startswith("\\"):
            sym = sym.lstrip("\\")
            if sym not in (_mathtext.Parser._overunder_functions |
                           _mathtext.Parser._function_names):
                sym = chr(_mathtext_data.tex2uni[sym])
        return f'\\{sym}' if sym in ('\\', '|', '+', '-', '*') else sym

    lines = []
    for category, syms in symbols:
        syms = sorted(syms,
                      # Sort by Unicode and place variants immediately
                      # after standard versions.
                      key=lambda sym: (render_symbol(sym, ignore_variant=True),
                                       sym.startswith(r"\var")),
                      reverse=(category == "Hebrew"))  # Hebrew is rtl
        rendered_syms = [f"{render_symbol(sym)} ``{sym}``" for sym in syms]
        columns = calculate_best_columns(
            rendered_syms,
            max_line_length=96,
            max_columns=6
        )
        lines.append("**%s**" % category)
        lines.append('')
        max_width = max(map(len, rendered_syms))
        header = (('=' * max_width) + ' ') * columns
        lines.append(header.rstrip())
        for part in range(0, len(rendered_syms), columns):
            row = " ".join(
                sym.rjust(max_width) for sym in rendered_syms[part:part + columns])
            lines.append(row)
        lines.append(header.rstrip())
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
    for category, syms in symbols:
        if category == "Standard Function Names":
            continue
        for sym in syms:
            if len(sym) > 1:
                all_symbols[sym[1:]] = None
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)

    # Add accents
    all_symbols.update({v[1:]: k for k, v in _mathtext.Parser._accent_map.items()})
    all_symbols.update({v: v for v in _mathtext.Parser._wide_accents})
    print("SYMBOLS NOT IN TABLE:")
    for sym, val in _mathtext_data.tex2uni.items():
        if sym not in all_symbols:
            print(f"{sym} = {chr(val)}")
