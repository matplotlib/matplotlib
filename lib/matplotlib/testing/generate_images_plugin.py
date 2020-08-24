import pytest


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--generate_images",
             action="store_true",
             default=False,
             help="run matplotlib baseline image generation tests"
    )
    except:
        pass


def pytest_collection_modifyitems(config, items):
    if config.getoption("--generate_images"):
        skip_non_baseline_img_gen_tests \
            = pytest.mark.skip(reason="No need to run non "
                                      "image generation tests")
        for item in items:
            if "generate_images" not in item.keywords:
                item.add_marker(skip_non_baseline_img_gen_tests)
