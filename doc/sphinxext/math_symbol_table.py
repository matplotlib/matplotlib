import re
from docutils.parsers.rst import Directive

from matplotlib import _mathtext, _mathtext_data
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
bb_pattern = re.compile(r'\\mathbb{([^}]+)}')
script_pattern = re.compile(r'\\mathscr{([^}]+)}')
fraktur_pattern = re.compile(r'\\mathfrak{([^}]+)}')

symbols =[
    ('Black-board characters', 4, ['A', 'B', 'C', 'D']),
    ('Script characters', 3, ['E', 'F', 'G', 'H', 'I', 'J']),
    ('Fraktur characters', 2, ['K', 'L', 'M', 'N', 'O'])
]


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
    for category, columns, syms in symbols:
        syms = sorted(syms,
                      # Sort by Unicode and place variants immediately
                      # after standard versions.
                      key=lambda sym: (render_symbol(sym, ignore_variant=True),
                                       sym.startswith(r"\var")),
                      reverse=(category == "Hebrew"))  # Hebrew is rtl
        rendered_syms = [f"{render_symbol(sym)} ``{sym}``" for sym in syms]
        columns = min(columns, len(syms))
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

def render_symbol(symbol):
    # Render symbol as Unicode character or escape sequence
    if symbol.startswith('\\'):
        return symbol.encode().decode('unicode_escape')
    return symbol

class MathSymbolTableDirective(Directive):
    def run(self):
        table_data = []
        for category, columns, syms in symbols:
            if not syms:
                continue
            table_data.append([f'**{category}**'])
            max_width = max(len(sym) for sym in syms)
            for i in range(0, len(syms), columns):
                row = syms[i:i + columns]
                row += [''] * (columns - len(row))  # Fill empty cells if needed
                table_data.append([f'``{render_symbol(sym)}``' for sym in row])
            table_data.append(['=' * max_width] * columns)
            table_data.append([])  # Add empty row between categories
        table_node = plt.table(cellText=table_data, loc='center')
        plt.axis('off')
        return table_node,

def setup(app):
    app.add_directive('mathsymboltable', MathSymbolTableDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


# Verify symbols exist in Stix font and print symbol information
font = FontProperties(family='STIXGeneral')

for _, _, syms in symbols:
    for sym in syms:
        exists = font.has_char(ord(render_symbol(sym)))
        print(f'Symbol: {sym}, Exists: {exists}')

# Test the MathSymbolTableDirective
from docutils.core import publish_string

rst_content = """
    .. mathsymboltable::

    """

publish_string(rst_content, writer_name='html')

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

    # Add accents
    all_symbols.update({v[1:]: k for k, v in _mathtext.Parser._accent_map.items()})
    all_symbols.update({v: v for v in _mathtext.Parser._wide_accents})
    print("SYMBOLS NOT IN TABLE:")
    for sym, val in _mathtext_data.tex2uni.items():
        if sym not in all_symbols:
            print(f"{sym} = {chr(val)}")
