#!/usr/bin/env python3
import argparse
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile

PREFIX = "/opt/homebrew"
MARKER = ".brew-delta"
MANIFEST = "/tmp/brew-manifest.json"
TOP_N = 20


class Counter(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.count = 0

    def writable(self):
        return True

    def write(self, data):
        self.count += len(data)
        return len(data)


def run(cmd):
    subprocess.run(cmd, check=True)


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def file_size(entry):
    return entry[2] if entry[0] == "f" else 0


def compressed_size(tree):
    counter = Counter()
    with tarfile.open(fileobj=counter, mode="w|gz") as tar:
        tar.add(tree, arcname=".")
    return counter.count


def scan(root):
    entries = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
                if stat.S_ISLNK(st.st_mode):
                    entry = ["l", os.readlink(path)]
                elif stat.S_ISDIR(st.st_mode):
                    entry = ["d", st.st_mode]
                else:
                    entry = ["f", st.st_mode, st.st_size, st.st_mtime_ns]
            except FileNotFoundError:
                # File removed while scanning
                continue
            entries[os.path.relpath(path, root)] = entry
    return entries


def copy_entry(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.isfile(dst):
        os.remove(dst)
    st = os.lstat(src)
    if stat.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(src), dst)
    elif stat.S_ISDIR(st.st_mode):
        os.makedirs(dst, exist_ok=True)
        shutil.copystat(src, dst, follow_symlinks=False)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def stage(rel, out):
    parent = os.path.dirname(rel)
    if parent and not os.path.lexists(os.path.join(out, parent)):
        stage(parent, out)
    copy_entry(os.path.join(PREFIX, rel), os.path.join(out, rel))


def write_report(path, added, modified, deleted, before, after, tree):
    size_added = sum(file_size(after[p]) for p in added)
    size_modified = sum(file_size(after[p]) for p in modified)
    size_removed = sum(file_size(before[p]) for p in deleted)
    payload = size_added + size_modified
    packed = compressed_size(tree)
    largest = sorted(
        added + modified, key=lambda p: file_size(after[p]), reverse=True
    )[:TOP_N]
    lines = [
        "SIZE REPORT",
        f"added:    {len(added)} entries, {human(size_added)}",
        f"modified: {len(modified)} entries, {human(size_modified)}",
        f"removed:  {len(deleted)} entries, {human(size_removed)}",
        f"cache payload (added + modified): {human(payload)}",
        f"cache payload gzip estimate: {human(packed)}",
        "",
        f"LARGEST {TOP_N} CHANGED FILES",
    ]
    lines += [f"  {human(file_size(after[p])):>10}  {p}" for p in largest]
    lines += ["", "ADDED"]
    lines += [f"  {p} ({human(file_size(after[p]))})" for p in added]
    lines += ["", "MODIFIED"]
    lines += [f"  {p} ({human(file_size(after[p]))})" for p in modified]
    lines += ["", "REMOVED"]
    lines += [f"  {p} ({human(file_size(before[p]))})" for p in deleted]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_snapshot(args):
    with open(MANIFEST, "w") as f:
        json.dump(scan(PREFIX), f)


def cmd_diff(args):
    with open(MANIFEST) as f:
        before = json.load(f)
    after = scan(PREFIX)
    added = sorted(p for p in after if p not in before)
    modified = sorted(p for p in after if p in before and before[p] != after[p])
    deleted = sorted(p for p in before if p not in after)
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(os.path.join(args.out, MARKER))
    for rel in added + modified:
        stage(rel, args.out)
    with open(os.path.join(args.out, MARKER, "deleted.txt"), "w") as f:
        f.write("\n".join(deleted))
    shutil.copystat(PREFIX, args.out)
    write_report(args.log, added, modified, deleted, before, after, args.out)


def cmd_finalize(args):
    marker = os.path.join(PREFIX, MARKER)
    with open(os.path.join(marker, "deleted.txt")) as f:
        deleted = [line for line in f.read().split("\n") if line]
    for rel in deleted:
        path = os.path.join(PREFIX, rel)
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    shutil.rmtree(marker)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    diff = sub.add_parser("diff")
    diff.add_argument("out")
    diff.add_argument("--log", default="brew-delta.log")
    sub.add_parser("swap-in").add_argument("out")
    sub.add_parser("swap-out")
    sub.add_parser("finalize")
    args = parser.parse_args()
    handlers = {
        "snapshot": cmd_snapshot,
        "diff": cmd_diff,
        "finalize": cmd_finalize,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
