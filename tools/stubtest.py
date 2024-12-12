import ast
import os
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).parent.parent

lib = root / "lib"
mpl = lib / "matplotlib"


class Visitor(ast.NodeVisitor):
    def __init__(self, filepath, output, existing_allowed):
        self.filepath = filepath
        self.context = list(filepath.with_suffix("").relative_to(lib).parts)
        self.output = output
        self.existing_allowed = existing_allowed

    def _is_already_allowed(self, parts):
        # Skip outputting a path if it's already allowed before.
        candidates = ['.'.join(parts[:s]) for s in range(1, len(parts))]
        for allow in self.existing_allowed:
            if any(allow.fullmatch(path) for path in candidates):
                return True
        return False

    def visit_FunctionDef(self, node):
        # delete_parameter adds a private sentinel value that leaks
        # we do not want that sentinel value in the type hints but it breaks typing
        # Does not apply to variadic arguments (args/kwargs)
        for dec in node.decorator_list:
            if "delete_parameter" in ast.unparse(dec):
                deprecated_arg = dec.args[1].value
                if (
                    node.args.vararg is not None
                    and node.args.vararg.arg == deprecated_arg
                ):
                    continue
                if (
                    node.args.kwarg is not None
                    and node.args.kwarg.arg == deprecated_arg
                ):
                    continue

                parents = []
                if hasattr(node, "parent"):
                    parent = node.parent
                    while hasattr(parent, "parent") and not isinstance(
                        parent, ast.Module
                    ):
                        parents.insert(0, parent.name)
                        parent = parent.parent
                parts = [*self.context, *parents, node.name]
                if not self._is_already_allowed(parts):
                    self.output.write("\\.".join(parts) + "\n")
                break

    def visit_ClassDef(self, node):
        for dec in node.decorator_list:
            if "define_aliases" in ast.unparse(dec):
                parents = []
                if hasattr(node, "parent"):
                    parent = node.parent
                    while hasattr(parent, "parent") and not isinstance(
                        parent, ast.Module
                    ):
                        parents.insert(0, parent.name)
                        parent = parent.parent
                aliases = ast.literal_eval(dec.args[0])
                # Written as a regex rather than two lines to avoid unused entries
                # for setters on items with only a getter
                for substitutions in aliases.values():
                    parts = self.context + parents + [node.name]
                    for a in substitutions:
                        if not (self._is_already_allowed([*parts, f"get_{a}"]) and
                                self._is_already_allowed([*parts, f"set_{a}"])):
                            self.output.write("\\.".join([*parts, f"[gs]et_{a}\n"]))
        for child in ast.iter_child_nodes(node):
            self.visit(child)


existing_allowed = []
with (root / 'ci/mypy-stubtest-allowlist.txt').open() as f:
    for line in f:
        line, _, _ = line.partition('#')
        line = line.strip()
        if line:
            existing_allowed.append(re.compile(line))


with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "allowlist.txt"
    with p.open("wt") as f:
        for path in mpl.glob("**/*.py"):
            v = Visitor(path, f, existing_allowed)
            tree = ast.parse(path.read_text())

            # Assign parents to tree so they can be backtraced
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            v.visit(tree)
    proc = subprocess.run(
        [
            "stubtest",
            "--mypy-config-file=pyproject.toml",
            "--allowlist=ci/mypy-stubtest-allowlist.txt",
            f"--allowlist={p}",
            "matplotlib",
        ],
        cwd=root,
        env=os.environ | {"MPLBACKEND": "agg"},
    )
    try:
        os.unlink(f.name)
    except OSError:
        pass

sys.exit(proc.returncode)
