from pathlib import Path
import os


if bp_str := os.environ.get("MPLTESTIMAGEPATH", None):
    base_path = (
        Path(bp_str) / 'mpl_toolkits' / 'axes_grid1' / 'tests' / 'baseline_images'
    )
else:
    base_path = Path(__file__).parent / 'baseline_images'


# Check that the test directories exist.
if not (base_path).exists() and not os.environ.get("MPLGENERATEBASELINE"):
    raise OSError(
        f'The baseline image directory ({base_path!r}) does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')

del base_path
del bp_str
