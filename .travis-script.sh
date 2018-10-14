#!/bin/bash
set -e
# fail immediately if one of the following commands fails

if [[ $RUN_FLAKE8 == 1 ]]; then
    flake8 --statistics && echo "Flake8 passed without any issues!"
fi

echo "Calling pytest with the following arguments: $PYTEST_ADDOPTS"
python -mpytest
