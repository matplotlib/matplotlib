from pathlib import Path
import os


if (base_path := os.environ.get("MPLTESTIMAGEPATH", None)) is None:
    base_path = Path(__file__).parent
else:
    base_path = Path(base_path) / 'matplotlib' / 'tests'

# Check that the test directories exist.
if not (base_path / 'baseline_images').exists():
    raise OSError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')
