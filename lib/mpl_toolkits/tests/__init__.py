from pathlib import Path

try:
    import matplotlib_baseline_images
except:
    if not (Path(__file__).parent / "baseline_images").exists():
        raise ImportError(
            'The baseline image directory does not exist. '
            'This is most likely because the test data is not installed '
            'and baseline image directory is absent. You may need to '
            'install matplotlib_baseline_images to get the test data.'
        )
