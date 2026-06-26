# Per-release pinned requirements

This directory contains fully pinned requirements files for each active
Matplotlib bug-fix (backport) branch.

## Naming convention

Files are named `fully-pinned-vX.Y.x.txt`, where `X.Y` matches the
release branch (e.g. `v3.11.x`).

## Workflow

When a new bug-fix branch `vX.Y.x` is created from `main`:
1. Copy the current `fully-pinned-vMAIN.txt` (or generate a fresh one)
   and rename it to `fully-pinned-vX.Y.x.txt`.
2. Pin all versions to the exact releases valid for that branch.
3. Use this file exclusively in CI jobs on the backport branch to
   ensure reproducible builds and reduce dependency-only backports.

## Updating

If a dependency needs a security patch on a backport branch, update the
pinned version in the corresponding file and open a PR targeting that branch.
