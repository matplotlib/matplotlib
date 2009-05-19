#!/bin/bash
set -e

TARGET_DIR="diff-images"
rm -rf $TARGET_DIR
mkdir $TARGET_DIR
find . -name 'failed-diff-*png' | xargs mv --target-directory=$TARGET_DIR
