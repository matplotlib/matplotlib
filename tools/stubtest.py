import ast
import os
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).parent.parent

lib = root / "lib"
mpl = lib / "matplotlib"


class Visitor(ast.NodeVisitor):
    def __init__(self, filepath, output):
        self.filepath = filepath
        self.context = list(filepath.with_suffix("").relative_to(lib).parts)
        self.output = output

    def visit_FunctionDef(self, node):
        if any("delete_parameter" in ast.unparse(line) for line in node.decorator_list):
            parents = []
            if hasattr(node, "parent"):
                parent = node.parent
                while hasattr(parent, "parent") and not isinstance(parent, ast.Module):
                    parents.insert(0, parent.name)
                    parent = parent.parent
            self.output.write(f"{'.'.join(self.context + parents)}.{node.name}\n")

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
                    self.output.write(
                        "\n".join(
                            f"{'.'.join(parts)}.[gs]et_{a}\n" for a in substitutions
                        )
                    )
        for child in ast.iter_child_nodes(node):
            self.visit(child)


with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "allowlist.txt"
    with p.open("wt") as f:
        for path in mpl.glob("**/*.py"):
            v = Visitor(path, f)
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
