#!/usr/bin/env python3

"""
Check that all specified .whl files have the correct LICENSE files included.

To run:
    $ python3 -m build --wheel
    $ ./ci/check_wheel_licenses.py dist/*.whl
"""

from pathlib import Path
import sys
import zipfile


if len(sys.argv) <= 1:
    sys.exit('At least one wheel must be specified in command-line arguments.')

project_dir = Path(__file__).parent.resolve().parent
license_dir = project_dir / 'LICENSE'

license_file_names = {path.name for path in sorted(license_dir.glob('*'))}
for wheel in sys.argv[1:]:
    print(f'Checking LICENSE files in: {wheel}')
    with zipfile.ZipFile(wheel) as f:
        wheel_license_file_names = {Path(path).name
                                    for path in sorted(f.namelist())
                                    if '.dist-info/LICENSE' in path}
        if not (len(wheel_license_file_names) and
                wheel_license_file_names.issuperset(license_file_names)):
            sys.exit(f'LICENSE file(s) missing:\n'
                     f'{wheel_license_file_names} !=\n'
                     f'{license_file_names}')
