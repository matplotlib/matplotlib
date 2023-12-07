"""Shared constants and functions."""

from typing import List, Optional, Sequence

from docutils import nodes
from docutils.parsers.rst import directives

WARNING_TYPE = "design"

SEMANTIC_COLORS = (
    "primary",
    "secondary",
    "success",
    "info",
    "warning",
    "danger",
    "light",
    "muted",
    "dark",
    "white",
    "black",
)


def create_component(
    name: str,
    classes: Sequence[str] = (),
    *,
    rawtext: str = "",
    children: Sequence[nodes.Node] = (),
    **attributes,
) -> nodes.container:
    """Create a container node for a design component."""
    node = nodes.container(
        rawtext, is_div=True, design_component=name, classes=list(classes), **attributes
    )
    node.extend(children)
    return node


def is_component(node: nodes.Node, name: str):
    """Check if a node is a certain design component."""
    try:
        return node.get("design_component") == name
    except AttributeError:
        return False


def make_choice(choices: Sequence[str]):
    """Create a choice validator."""
    return lambda argument: directives.choice(argument, choices)


def _margin_or_padding_option(
    argument: Optional[str],
    class_prefix: str,
    allowed: Sequence[str],
) -> List[str]:
    """Validate the margin/padding is one (all) or four (top bottom left right) integers,
    between 0 and 5 or 'auto'.
    """
    if argument is None:
        raise ValueError("argument required but none supplied")
    values = argument.split()
    for value in values:
        if value not in allowed:
            raise ValueError(f"{value} is not in: {allowed}")
    if len(values) == 1:
        return [f"{class_prefix}-{values[0]}"]
    if len(values) == 4:
        return [
            f"{class_prefix}{side}-{value}"
            for side, value in zip(["t", "b", "l", "r"], values)
        ]
    raise ValueError(
        "argument must be one (all) or four (top bottom left right) integers"
    )


def margin_option(argument: Optional[str]) -> List[str]:
    """Validate the margin is one (all) or four (top bottom left right) integers,
    between 0 and 5 or 'auto'.
    """
    return _margin_or_padding_option(
        argument, "sd-m", ("auto", "0", "1", "2", "3", "4", "5")
    )


def padding_option(argument: Optional[str]) -> List[str]:
    """Validate the padding is one (all) or four (top bottom left right) integers,
    between 0 and 5.
    """
    return _margin_or_padding_option(argument, "sd-p", ("0", "1", "2", "3", "4", "5"))


def text_align(argument: Optional[str]) -> List[str]:
    """Validate the text align is left, right, center or justify."""
    value = directives.choice(argument, ["left", "right", "center", "justify"])
    return [f"sd-text-{value}"]


class PassthroughTextElement(nodes.TextElement):
    """A text element which will not render anything.

    This is required for reference node to render correctly outside of paragraphs.
    Since sphinx expects them to be within a ``TextElement``:
    https://github.com/sphinx-doc/sphinx/blob/068f802df90ea790f89319094e407c4d5f6c26ff/sphinx/writers/html5.py#L224
    """
