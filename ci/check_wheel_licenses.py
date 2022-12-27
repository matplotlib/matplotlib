#!/usr/bin/env python3

"""
Check that all .whl files in the dist folder have the correct LICENSE files
included.

To run:
    $ python3 setup.py bdist_wheel
    $ ./ci/check_wheel_licenses.py
"""

from pathlib import Path
import sys
import zipfile

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

project_dir = Path(__file__).parent.resolve().parent
dist_dir = project_dir / 'dist'
license_dir = project_dir / 'LICENSE'

license_file_names = {path.name for path in sorted(license_dir.glob('*'))}
for wheel in dist_dir.glob('*.whl'):
    print(f'Checking LICENSE files in: {wheel}')
    with zipfile.ZipFile(wheel) as f:
        wheel_license_file_names = {Path(path).name
                                    for path in sorted(f.namelist())
                                    if '.dist-info/LICENSE' in path}
        if not (len(wheel_license_file_names) and
                wheel_license_file_names.issuperset(license_file_names)):
            print(f'LICENSE file(s) missing:\n'
                  f'{wheel_license_file_names} !=\n'
                  f'{license_file_names}')
            sys.exit(EXIT_FAILURE)
sys.exit(EXIT_SUCCESS)
