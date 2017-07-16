#!/bin/bash

set -e

if [ "$CIRCLE_BRANCH" != "master" -o "$CIRCLE_PULL_REQUEST" != "" ]; then
    echo "Not uploading docs from non-master branch."
    exit
fi

git clone git@github.com:matplotlib/devdocs.git

cd devdocs

git checkout --orphan gh-pages || true
git reset --hard first_commit

cp -R ../doc/build/html/* .
touch .nojekyll

git config user.email "MatplotlibCircleBot@nomail"
git config user.name "MatplotlibCircleBot"
git config push.default simple

git add .
git commit -m "Docs build of $CIRCLE_SHA1"

git push --set-upstream origin gh-pages --force
