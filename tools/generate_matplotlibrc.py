#!/usr/bin/env python
"""
Generate matplotlirc for installs.

If packagers want to change the default backend, insert a `#backend: ...` line.
Otherwise, use the default `##backend: Agg` which has no effect even after
decommenting, which allows _auto_backend_sentinel to be filled in at import
time.
"""

import sys
from pathlib import Path


if len(sys.argv) != 4:
    raise SystemExit('usage: {sys.argv[0]} <input> <output> <backend>')

input = Path(sys.argv[1])
output = Path(sys.argv[2])
backend = sys.argv[3]

template_lines = input.read_text(encoding="utf-8").splitlines(True)
backend_line_idx, = [  # Also asserts that there is a single such line.
    idx for idx, line in enumerate(template_lines)
    if "#backend:" in line]
template_lines[backend_line_idx] = (
    f"#backend: {backend}\n" if backend not in ['', 'auto'] else "##backend: Agg\n")
output.write_text("".join(template_lines), encoding="utf-8")
