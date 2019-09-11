import ast
import inspect
import logging
from pathlib import Path
import tokenize

import matplotlib
from matplotlib import cbook

_log = logging.getLogger(__name__)


def node_match(node, name, **kwargs):
    return (type(node).__name__ == name
            and all(getattr(node, k) == v for k, v in kwargs.items()))


def bind_arguments(func, node, qualname):
    ba = inspect.signature(func).bind(
        *node.args, **{k.arg: k.value for k in node.keywords})
    for k, v in [*ba.arguments.items()]:
        if node_match(v, "NameConstant"):
            ba.arguments[k] = v.value
        if node_match(v, "Num"):
            ba.arguments[k] = v.n
        elif node_match(v, "Str"):
            ba.arguments[k] = v.s
        else:
            _log.warning(f"In {qualname} line {node.lineno}, "
                         f"unsupported argument: {v}")
    return ba


def generate_apichanges_deprecated(name, arguments):
    obj_type = arguments["obj_type"]
    s = [f"The {name} {obj_type} is deprecated."]
    if arguments["alternative"]:
        s += [f"Use {arguments['alternative']} instead."]
    return "  ".join(s)


def generate_apichanges_warn_deprecated(name, arguments):
    try:
        return cbook.deprecation._generate_deprecation_warning(**arguments)
    except TypeError:
        return "FIXME"


class DeprecationDetector(ast.NodeVisitor):
    def __init__(self, module_name):
        super().__init__()
        self._qualname_parts = [module_name]
        self.deprecateds = []
        self.warn_deprecateds = []

    current_qualname = property(lambda self: ".".join(self._qualname_parts))

    def _visit_class_or_function(self, node):
        self._qualname_parts.append(node.name)
        obj_type = {"ClassDef": "class", "FunctionDef": "function"}[
            type(node).__name__]
        for decorator in node.decorator_list[::-1]:
            if node_match(decorator, "Name", id="property"):
                obj_type = "property"
            if node_match(decorator, "Call"):
                func = decorator.func
                if (node_match(func, "Name", id="deprecated")
                        or node_match(func, "Attribute", attr="deprecated")):
                    ba = bind_arguments(cbook.deprecated, decorator,
                                        self.current_qualname)
                    ba.arguments.setdefault("obj_type", obj_type)
                    ba.apply_defaults()
                    self.deprecateds.append(
                        (self.current_qualname, ba.arguments))
        super().generic_visit(node)
        self._qualname_parts.pop()

    visit_ClassDef = visit_FunctionDef = _visit_class_or_function

    def visit_Call(self, node):
        if (node_match(node.func, "Name", id="warn_deprecated")
                or node_match(node.func, "Attribute", attr="warn_deprecated")):
            ba = bind_arguments(cbook.warn_deprecated, node,
                                self.current_qualname)
            self.warn_deprecateds.append((self.current_qualname, ba.arguments))


def main():
    root = Path(matplotlib.__file__).parent.parent
    deprecateds = []
    warn_deprecateds = []
    for path in sorted(root.glob("**/*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = "".join(f".{part}" for part in parts[1:])  # Drop pkg name.
        with tokenize.open(path) as file:
            source = file.read()
        tree = ast.parse(source)
        dd = DeprecationDetector(module_name)
        dd.visit(tree)
        deprecateds.extend(dd.deprecateds)
        warn_deprecateds.extend(dd.warn_deprecateds)
    by_version = {}
    for name, arguments in deprecateds:
        by_version.setdefault(arguments["since"], []).append((name, arguments))
    for version, deprecateds in sorted(by_version.items()):
        print()
        print(f"Since {version}:")
        for name, arguments in deprecateds:
            print(generate_apichanges_deprecated(name, arguments))
    by_version = {}
    for name, arguments in warn_deprecateds:
        by_version.setdefault(arguments["since"], []).append((name, arguments))
    # for version, warn_deprecateds in sorted(by_version.items()):
    for version, warn_deprecateds in by_version.items():
        print()
        print(f"Since {version}:")
        for name, arguments in warn_deprecateds:
            print(generate_apichanges_warn_deprecated(name, arguments))


if __name__ == "__main__":
    main()
