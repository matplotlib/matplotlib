import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
            "--generate-images",
             action="store_true",
             default=bool(os.environ.get('MPLGENERATEBASELINE', False)),
             help="run matplotlib baseline image generation tests"
    )
    parser.addoption(
            "--image-baseline",
             default=os.environ.get('MPLBASELINEIMAGES', None),
             help="run matplotlib baseline image generation tests",
        type=str
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--generate-images"):
        skip_non_matplotlib_baseline_image_generation_tests = pytest.mark.skip(
            reason="No need to run non image generation tests"
        )
        for item in items:
            if "generate_images" not in item.keywords:
                item.add_marker(skip_non_matplotlib_baseline_image_generation_tests)
