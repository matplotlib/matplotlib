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

        style = dedent(
            "\n".join(
                [
                    "<style>",
                    ".mpl-symbol-table { margin: 0 0 1rem 0; }",
                    ".mpl-symbol-grid {",
                    "  display: grid;",
                    "  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));",
                    "  gap: 0.5rem 1rem;",
                    "  align-items: center;",
                    "}",
                    ".mpl-symbol-cell {",
                    "  display: flex;",
                    "  align-items: center;",
                    "  gap: 0.6rem;",
                    "  padding: 0.2rem 0.1rem;",
                    "  white-space: nowrap;",
                    "}",
                    ".mpl-symbol-cell .label {",
                    "  font-family: monospace;",
                    "  font-size: 0.9em;",
                    "  color: #333;",
                    "}",
                    ".mpl-symbol-cell .math {",
                    "  font-size: 1.05em;",
                    "}",
                    "</style>",
                ]
            )
        )

        # Insert the style as raw HTML block
        lines.append(".. raw:: html")
        lines.append("")
        for style_line in style.splitlines():
            lines.append("   " + style_line)
        lines.append("")

        # Get symbol categories from matplotlib mathtext internals.
        try:
            categories = _mathtext._get_sphinx_symbol_table()
        except Exception:
            categories = []

        for category, _, syms in categories:
            # Ensure consistent ordering for reproducible output.
            syms_list = sorted(list(syms), key=lambda s: str(s))

            lines.append(f"**{category}**")
            lines.append("")
            lines.append(".. raw:: html")
            lines.append("")
            lines.append('   <div class="mpl-symbol-table">')
            lines.append('     <div class="mpl-symbol-grid">')

            for sym in syms_list:
                s = str(sym)
                # Use raw TeX inside \( ... \) so MathJax (Sphinx) renders it.
                tex = s
                html_line = (
                    "       <div class=\"mpl-symbol-cell\">"
                    f"<span class=\"math\">\\({tex}\\)</span>"
                    f"<span class=\"label\">`{s}`</span>"
                    "</div>"
                )
                lines.append(html_line)

            lines.append("     </div>")
            lines.append("   </div>")
            lines.append("")

        # Let Sphinx parse the lines so roles and references work.
        text = "\n".join(lines)
        node = nodes.paragraph()
        self.state.nested_parse(StringList(text.splitlines()), 0, node)
        return [node]


def setup(app):
    """Register the Sphinx directive."""
    app.add_directive("math_symbol_table", MathSymbolTableDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
