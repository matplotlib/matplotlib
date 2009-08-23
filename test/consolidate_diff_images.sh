#!/bin/bash
set -e

TARGET_DIR="diff-images"
rm -rf $TARGET_DIR
mkdir $TARGET_DIR
find . -name 'failed-diff-*png' -exec mv {} $TARGET_DIR/ \;
