# Fully pinned CI requirements

This directory contains fully pinned requirements files for Matplotlib CI.
The files pin transitive dependencies for one Linux CI target so that
bug-fix branches can keep using a stable dependency set.

## Naming convention

- `fully-pinned-main.txt` is used by the pinned-requirements job on `main`.
- `fully-pinned-vX.Y.x.txt` is used by jobs on the matching bug-fix branch.
- Matching `.in` files contain the direct requirements used to regenerate
  the pinned `.txt` files.

## Workflow

When a new bug-fix branch `vX.Y.x` is created from `main`:
1. Copy `fully-pinned-main.in` to `fully-pinned-vX.Y.x.in`.
2. Adjust the input file to match the branch's supported Python version
   and dependency ranges.
3. Generate `fully-pinned-vX.Y.x.txt`, for example:

   ```bash
   uv pip compile \
     --python-version 3.11 \
     --python-platform x86_64-manylinux2014 \
     --output-file ci/requirements/fully-pinned-v3.11.x.txt \
     ci/requirements/fully-pinned-v3.11.x.in
   ```

4. Backport the generated `.in` and `.txt` files to the bug-fix branch.
   The test workflow automatically uses `ci/requirements/fully-pinned-${branch}.txt`
   on non-`main` branches when it exists.

## Updating

If a dependency needs a security patch on a backport branch, update the
pinned version by regenerating the corresponding `.txt` file from its `.in`
file and open a PR targeting that branch.
