#!/usr/bin/env python3
"""
Download YAML Schemas for linting and validation.

Since pre-commit CI doesn't have Internet access, we need to bundle these files
in the repo.
"""

import os
import pathlib
import urllib.request


HERE = pathlib.Path(__file__).parent
SCHEMAS = [
    'https://json.schemastore.org/appveyor.json',
    'https://json.schemastore.org/circleciconfig.json',
    'https://json.schemastore.org/github-funding.json',
    'https://json.schemastore.org/github-issue-config.json',
    'https://json.schemastore.org/github-issue-forms.json',
    'https://json.schemastore.org/codecov.json',
    'https://json.schemastore.org/pull-request-labeler-5.json',
    'https://github.com/microsoft/vscode-python/raw/'
        'main/schemas/conda-environment.json',
]


def print_progress(block_count, block_size, total_size):
    size = block_count * block_size
    if total_size != -1:
        size = min(size, total_size)
        width = 50
        percent = size / total_size * 100
        filled = int(percent // (100 // width))
        percent_str = '\N{Full Block}' * filled + '\N{Light Shade}' * (width - filled)
    print(f'{percent_str} {size:6d} / {total_size:6d}', end='\r')


# First clean up existing files.
for json in HERE.glob('*.json'):
    os.remove(json)

for schema in SCHEMAS:
    path = HERE / schema.rsplit('/', 1)[-1]
    print(f'Downloading {schema} to {path}')
    urllib.request.urlretrieve(schema, filename=path, reporthook=print_progress)
    print()
    # This seems weird, but it normalizes line endings to the current platform,
    # so that Git doesn't complain about it.
    path.write_text(path.read_text())
