"""
Sphinx extension that generates the math symbol table documentation
for Matplotlib.
"""
from __future__ import annotations
from textwrap import dedent
from docutils.statemachine import StringList
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from matplotlib import _mathtext


class MathSymbolTableDirective(SphinxDirective):
    """Generate tables of math symbols grouped by category."""

    has_content = False

    def run(self):
        # Build RST lines to be parsed. We include a small CSS style and
        # simple HTML wrappers so the result is responsive in the browser.
        lines: list[str] = []

        # Add responsive CSS styling
        style = dedent(
            "\n".join(
                [
                    "",
                    "",
                    ".mpl-symbol-grid {",
                    "    display: grid;",
                    "    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));",
                    "    gap: 8px;",
                    "    margin: 16px 0;",
                    "}",
                    ".mpl-symbol-cell {",
                    "    display: flex;",
                    "    flex-direction: column;",
                    "    align-items: center;",
                    "    padding: 8px;",
                    "    border: 1px solid #ddd;",
                    "    border-radius: 4px;",
                    "}",
                    ".mpl-symbol-cell .math {",
                    "    font-size: 1.2em;",
                    "    margin-bottom: 4px;",
                    "}",
                    ".mpl-symbol-cell .label {",
                    "    font-family: monospace;",
                    "    font-size: 0.9em;",
                    "    color: #666;",
                    "}",
                    "",
                    "",
                ]
            )
        )

        # Insert the style as raw HTML block
        lines.append(".. raw:: html")
        lines.append("")
        for style_line in style.splitlines():
            lines.append("   " + style_line)
        lines.append("")

        # Get symbol categories from matplotlib mathtext internals
        try:
            categories = _mathtext._get_sphinx_symbol_table()
        except Exception:
            categories = []

        for category, _, syms in categories:
            # Ensure consistent ordering for reproducible output
            syms_list = sorted(list(syms), key=lambda s: str(s))

            lines.append(f"**{category}**")
            lines.append("")
            lines.append(".. raw:: html")
            lines.append("")
            lines.append('   <div class="mpl-symbol-grid" node="_65">')
            lines.append('   ')

            for sym in syms_list:
                s = str(sym)
                # Use raw TeX inside \( ... \) so MathJax (Sphinx) renders it
                tex = s
                html_line = (
                    '   <div class="mpl-symbol-cell" node="_74">'
                    f'<span class="math" node="_76">\\({tex}\\)</span>'
                    f'<span class="label" node="_78">`{s}`</span>'
                    "</div>"
                )
                lines.append(html_line)

            lines.append("   </div>")
            lines.append("   ")
            lines.append("")

        # Let Sphinx parse the lines so roles and references work
        text = "\n".join(lines)
        node = nodes.paragraph()
        self.state.nested_parse(StringList(text.splitlines()), 0, node)
        return [node]


def setup(app):
    """Register the Sphinx directive."""
    app.add_directive("math_symbol_table", MathSymbolTableDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
