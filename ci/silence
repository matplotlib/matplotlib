#!/bin/bash

# Run a command, hiding its standard output and error if its exit
# status is zero.

stdout=$(mktemp -t stdout) || exit 1
stderr=$(mktemp -t stderr) || exit 1
"$@" >$stdout 2>$stderr
code=$?
if [[ $code != 0 ]]; then
    cat $stdout
    cat $stderr >&2
    exit $code
fi
