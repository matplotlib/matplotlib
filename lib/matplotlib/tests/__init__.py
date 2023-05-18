from pathlib import Path
import os


if (base_path_str := os.environ.get("MPLTESTIMAGEPATH", None)) is None:
    base_path = Path(__file__).parent / 'baseline_images'
else:
    base_path = Path(base_path_str) / 'matplotlib' / 'tests' / 'baseline_images'


# Check that the test directories exist.
if not (base_path).exists():
    raise OSError(
        f'The baseline image directory ({base_path!r}) does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')

del base_path
del base_path_str
