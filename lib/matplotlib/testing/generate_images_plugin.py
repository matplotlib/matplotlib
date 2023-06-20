import pytest


def pytest_addoption(parser):
    parser.addoption(
            "--generate-images",
             action="store_true",
             default=False,
             help="run matplotlib baseline image generation tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--generate-images"):
        skip_non_matplotlib_baseline_image_generation_tests = pytest.mark.skip(
            reason="No need to run non image generation tests"
        )
        for item in items:
            if "generate_images" not in item.keywords:
                item.add_marker(skip_non_matplotlib_baseline_image_generation_tests)
