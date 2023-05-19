import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib.testing.decorators as mtd


def _mod_to_path(libpath, mod):
    return Path(libpath) / Path(*mod.split("."))


def _find_imagelist(libpath, mod, imagelist_name="image_list.txt"):
    return Path(libpath) / Path(*mod.split(".")[:-1]) / imagelist_name


def _add(args):
    image_list_path = _find_imagelist(args.libpath, args.module)
    data = mtd._load_imagelist(image_list_path)
    fname = Path(args.module.split(".")[-1]) / args.fname
    if fname in data:
        raise RuntimeError("Trying to add as existing file, did you mean to use 'rev'?")
    data[fname] = {"rev": 0, "ts": time.time()}
    mtd._write_imagelist(data, target_file=image_list_path)


def _rev(args):
    image_list_path = _find_imagelist(args.libpath, args.module)
    data = mtd._load_imagelist(image_list_path)
    fname = Path(args.module.split(".")[-1]) / args.fname
    if fname not in data:
        raise RuntimeError(
            "Trying to rev a non-existing file, did you mean to use 'add'?"
        )
    data[fname]["rev"] += 1
    data[fname]["ts"] = time.time()
    mtd._write_imagelist(data, target_file=image_list_path)


def _validate(args):
    image_list_path = _find_imagelist(args.libpath, args.package + ".a")
    data = mtd._load_blame(image_list_path)
    json_path = (
        Path(args.baseline_path)
        / Path(*args.package.split("."))
        / "baseline_images"
        / "metadata.json"
    )
    with open(json_path) as fin:
        md = {Path(k): v for k, v in json.load(fin).items()}

    if extra := set(md) ^ set(data):
        # TODO good error messages about where the extra files are
        print(f"{extra=}")
        sys.exit(1)

    mismatch = set()
    for k in md:
        if md[k]["sha"] != data[k]["sha"]:
            mismatch.add(k)
    if mismatch:
        print(f"{mismatch=}")
        sys.exit(1)


if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser(prog="manage baseline images")
    parser.add_argument(
        "--libpath",
        help="Relative path to package source.",
        default="",
        required=False,
    )

    subparsers = parser.add_subparsers(help="sub-command help", dest="cmd")

    # create the parser for the "rev" command
    parser_rev = subparsers.add_parser("rev", help="Version rev a test file.")
    parser_rev.add_argument("module", type=str, help="The dotted name of the module.")
    parser_rev.add_argument(
        "fname", type=str, help="The (relative) name of the file to version rev."
    )

    # create the parser for the "add" command
    parser_add = subparsers.add_parser("add", help="Add a new baseline image.")
    parser_add.add_argument("module", type=str, help="The dotted name of the module.")
    parser_add.add_argument(
        "fname", type=str, help="The (relative) name of the file to version rev."
    )

    # create the parser for the "add" command
    parser_add = subparsers.add_parser("validate", help="Check if the baseline dir .")
    parser_add.add_argument(
        "package", type=str, help="The dotted name of the test (sub-)package."
    )
    parser_add.add_argument(
        "baseline_path",
        type=str,
        help="The dotted name of the test (sub-)package.",
    )

    # parse some argument lists
    args = parser.parse_args()

    {"add": _add, "rev": _rev, "validate": _validate}[args.cmd](args)
