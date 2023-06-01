#!/usr/bin/env python3

"""
Determine the name of the sdist and export to GitHub output named SDIST_NAME.

To run:
    $ python3 -m build --sdist
    $ ./ci/determine_sdist_name.py
"""
import os
from pathlib import Path
import sys


paths = [p.name for p in Path("dist").glob("*.tar.gz")]
if len(paths) != 1:
    sys.exit(f"Only a single sdist is supported, but found: {paths}")

print(paths[0])
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"SDIST_NAME={paths[0]}\n")
