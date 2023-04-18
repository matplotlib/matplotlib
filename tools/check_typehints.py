#!/usr/bin/env python
"""
Perform AST checks to validate consistency of type hints with implementation.

NOTE: in most cases ``stubtest`` (distributed as part of ``mypy``)  should be preferred

This script was written before the configuration of ``stubtest`` was well understood.
It still has some utility, particularly for checking certain deprecations or other
decorators which modify the runtime signature where you want the type hint to match
the python source rather than runtime signature, perhaps.

The basic kinds of checks performed are:

- Set of items defined by the stubs vs implementation
  - Missing stub: MISSING_STUB = 1
  - Missing implementation: MISSING_IMPL = 2
- Signatures of functions/methods
  - Positional Only Args: POS_ARGS = 4
  - Keyword or Positional Args: ARGS = 8
  - Variadic Positional Args: VARARG = 16
  - Keyword Only Args: KWARGS = 32
  - Variadic Keyword Only Args: VARKWARG = 64

There are some exceptions to when these are checked:

- Set checks (MISSING_STUB/MISSING_IMPL) only apply at the module level
  - i.e. not for classes
  - Inheritance makes the set arithmetic harder when only loading AST
  - Attributes also make it more complicated when defined in init
- Functions type hinted with ``overload`` are ignored for argument checking
  - Usually this means the implementation is less strict in signature but will raise
    if an invalid signature is used, type checking allows such errors to be caught by
    the type checker instead of at runtime.
- Private attribute/functions are ignored
  - Not expecting a type hint
  - applies to anything beginning, but not ending in ``__``
  - If ``__all__`` is defined, also applies to anything not in ``__all__``
- Deprecated methods are not checked for missing stub
  - Only applies to wholesale deprecation, not deprecation of an individual arg
  - Other kinds of deprecations (e.g. argument deletion or rename) the type hint should
    match the current python definition line still.
      - For renames, the new name is used
      - For deletions/make keyword only, it is removed upon expiry

Usage:

Currently there is not any argument handling/etc, so all configuration is done in
source.
Since stubtest has almost completely superseded this script, this is unlikely to change.

The categories outlined above can each be ignored, and ignoring multiple can be done
using the bitwise or (``|``) operator, e.g. ``ARGS | VARKWARG``.

This can be done globally or on a per file basis, by editing ``per_file_ignore``.
For the latter, the key is the Pathlib Path to the affected file, and the value is the
integer ignore.

Must be run from repository root:

``python tools/check_typehints.py``
"""

import ast
import pathlib
import sys

MISSING_STUB = 1
MISSING_IMPL = 2
POS_ARGS = 4
ARGS = 8
VARARG = 16
KWARGS = 32
VARKWARG = 64


def check_file(path, ignore=0):
    stubpath = path.with_suffix(".pyi")
    ret = 0
    if not stubpath.exists():
        return 0, 0
    tree = ast.parse(path.read_text())
    stubtree = ast.parse(stubpath.read_text())
    return check_namespace(tree, stubtree, path, ignore)


def check_namespace(tree, stubtree, path, ignore=0):
    ret = 0
    count = 0
    tree_items = set(
        i.name
        for i in tree.body
        if hasattr(i, "name") and (not i.name.startswith("_") or i.name.endswith("__"))
    )
    stubtree_items = set(
        i.name
        for i in stubtree.body
        if hasattr(i, "name") and (not i.name.startswith("_") or i.name.endswith("__"))
    )

    for item in tree.body:
        if isinstance(item, ast.Assign):
            tree_items |= set(
                i.id
                for i in item.targets
                if hasattr(i, "id")
                and (not i.id.startswith("_") or i.id.endswith("__"))
            )
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    tree_items |= set(i.id for i in target.elts)
        elif isinstance(item, ast.AnnAssign):
            tree_items |= {item.target.id}
    for item in stubtree.body:
        if isinstance(item, ast.Assign):
            stubtree_items |= set(
                i.id
                for i in item.targets
                if hasattr(i, "id")
                and (not i.id.startswith("_") or i.id.endswith("__"))
            )
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    stubtree_items |= set(i.id for i in target.elts)
        elif isinstance(item, ast.AnnAssign):
            stubtree_items |= {item.target.id}

    try:
        all_ = ast.literal_eval(ast.unparse(get_subtree(tree, "__all__").value))
    except ValueError:
        all_ = []

    if all_:
        missing = (tree_items - stubtree_items) & set(all_)
    else:
        missing = tree_items - stubtree_items

    deprecated = set()
    for item_name in missing:
        item = get_subtree(tree, item_name)
        if hasattr(item, "decorator_list"):
            if "deprecated" in [
                i.func.attr
                for i in item.decorator_list
                if hasattr(i, "func") and hasattr(i.func, "attr")
            ]:
                deprecated |= {item_name}

    if missing - deprecated and ~ignore & MISSING_STUB:
        print(f"{path}: {missing - deprecated} missing from stubs")
        ret |= MISSING_STUB
        count += 1

    non_class_or_func = set()
    for item_name in stubtree_items - tree_items:
        try:
            get_subtree(tree, item_name)
        except ValueError:
            pass
        else:
            non_class_or_func |= {item_name}

    missing_implementation = stubtree_items - tree_items - non_class_or_func
    if missing_implementation and ~ignore & MISSING_IMPL:
        print(f"{path}: {missing_implementation} in stubs and not source")
        ret |= MISSING_IMPL
        count += 1

    for item_name in tree_items & stubtree_items:
        item = get_subtree(tree, item_name)
        stubitem = get_subtree(stubtree, item_name)
        if isinstance(item, ast.FunctionDef) and isinstance(stubitem, ast.FunctionDef):
            err, c = check_function(item, stubitem, f"{path}::{item_name}", ignore)
            ret |= err
            count += c
        if isinstance(item, ast.ClassDef):
            # Ignore set differences for classes... while it would be nice to have
            # inheritance and attributes set in init/methods make both presence and
            # absence of nodes spurious
            err, c = check_namespace(
                item,
                stubitem,
                f"{path}::{item_name}",
                ignore | MISSING_STUB | MISSING_IMPL,
            )
            ret |= err
            count += c

    return ret, count


def check_function(item, stubitem, path, ignore):
    ret = 0
    count = 0

    # if the stub calls overload, assume it knows what its doing
    overloaded = "overload" in [
        i.id for i in stubitem.decorator_list if hasattr(i, "id")
    ]
    if overloaded:
        return 0, 0

    item_posargs = [a.arg for a in item.args.posonlyargs]
    stubitem_posargs = [a.arg for a in stubitem.args.posonlyargs]
    if item_posargs != stubitem_posargs and ~ignore & POS_ARGS:
        print(
            f"{path} {item.name} posargs differ: {item_posargs} vs {stubitem_posargs}"
        )
        ret |= POS_ARGS
        count += 1

    item_args = [a.arg for a in item.args.args]
    stubitem_args = [a.arg for a in stubitem.args.args]
    if item_args != stubitem_args and ~ignore & ARGS:
        print(f"{path} args differ for {item.name}: {item_args} vs {stubitem_args}")
        ret |= ARGS
        count += 1

    item_vararg = item.args.vararg
    stubitem_vararg = stubitem.args.vararg
    if ~ignore & VARARG:
        if (item_vararg is None) ^ (stubitem_vararg is None):
            if item_vararg:
                print(
                    f"{path} {item.name} vararg differ: "
                    f"{item_vararg.arg} vs {stubitem_vararg}"
                )
            else:
                print(
                    f"{path} {item.name} vararg differ: "
                    f"{item_vararg} vs {stubitem_vararg.arg}"
                )
            ret |= VARARG
            count += 1
        elif item_vararg is None:
            pass
        elif item_vararg.arg != stubitem_vararg.arg:
            print(
                f"{path} {item.name} vararg differ: "
                f"{item_vararg.arg} vs {stubitem_vararg.arg}"
            )
            ret |= VARARG
            count += 1

    item_kwonlyargs = [a.arg for a in item.args.kwonlyargs]
    stubitem_kwonlyargs = [a.arg for a in stubitem.args.kwonlyargs]
    if item_kwonlyargs != stubitem_kwonlyargs and ~ignore & KWARGS:
        print(
            f"{path} {item.name} kwonlyargs differ: "
            f"{item_kwonlyargs} vs {stubitem_kwonlyargs}"
        )
        ret |= KWARGS
        count += 1

    item_kwarg = item.args.kwarg
    stubitem_kwarg = stubitem.args.kwarg
    if ~ignore & VARKWARG:
        if (item_kwarg is None) ^ (stubitem_kwarg is None):
            if item_kwarg:
                print(
                    f"{path} {item.name} varkwarg differ: "
                    f"{item_kwarg.arg} vs {stubitem_kwarg}"
                )
            else:
                print(
                    f"{path} {item.name} varkwarg differ: "
                    f"{item_kwarg} vs {stubitem_kwarg.arg}"
                )
            ret |= VARKWARG
            count += 1
        elif item_kwarg is None:
            pass
        elif item_kwarg.arg != stubitem_kwarg.arg:
            print(
                f"{path} {item.name} varkwarg differ: "
                f"{item_kwarg.arg} vs {stubitem_kwarg.arg}"
            )
            ret |= VARKWARG
            count += 1

    return ret, count


def get_subtree(tree, name):
    for item in tree.body:
        if isinstance(item, ast.Assign):
            if name in [i.id for i in item.targets if hasattr(i, "id")]:
                return item
            for target in item.targets:
                if isinstance(target, ast.Tuple):
                    if name in [i.id for i in target.elts]:
                        return item
        if isinstance(item, ast.AnnAssign):
            if name == item.target.id:
                return item
        if not hasattr(item, "name"):
            continue
        if item.name == name:
            return item
    raise ValueError(f"no such item {name} in tree")


if __name__ == "__main__":
    out = 0
    count = 0
    basedir = pathlib.Path("lib/matplotlib")
    per_file_ignore = {
        # Edge cases for items set via `get_attr`, etc
        basedir / "__init__.py": MISSING_IMPL,
        # Base class has **kwargs, subclasses have more specific
        basedir / "ticker.py": VARKWARG,
        basedir / "layout_engine.py": VARKWARG,
    }
    for f in basedir.rglob("**/*.py"):
        err, c = check_file(f, ignore=0 | per_file_ignore.get(f, 0))
        out |= err
        count += c
    print("\n")
    print(f"{count} total errors found")
    sys.exit(out)
