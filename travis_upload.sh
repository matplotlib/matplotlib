#!/bin/bash

set -e

echo "Compressing results"
tar cjf result_images.tar.bz2 result_images
echo "Uploading results"
travis-artifacts upload --path result_images.tar.bz2
echo "Results available at:"
echo https://s3.amazonaws.com/matplotlib-test-results/artifacts/${TRAVIS_BUILD_NUMBER}/${TRAVIS_JOB_NUMBER}/result_images.tar.bz2
