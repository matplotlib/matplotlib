import ast
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
                    parents.append(parent.name)
                    parent = parent.parent
            self.output.write(f"{'.'.join(self.context + parents)}.{node.name}\n")


with tempfile.NamedTemporaryFile("wt") as f:
    for path in mpl.glob("**/*.py"):
        v = Visitor(path, f)
        tree = ast.parse(path.read_text())

        # Assign parents to tree so they can be backtraced
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        v.visit(tree)
    f.flush()
    proc = subprocess.run(
        [
            "stubtest",
            "--mypy-config-file=pyproject.toml",
            "--allowlist=ci/mypy-stubtest-allowlist.txt",
            f"--allowlist={f.name}",
            "matplotlib",
        ],
        cwd=root,
    )

sys.exit(proc.returncode)
