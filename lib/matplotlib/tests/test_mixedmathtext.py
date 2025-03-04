import pytest
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties

@pytest.mark.parametrize("usetex", [False, True])  # Test both MathText and LaTeX modes
def test_mixed_math_text_x_alignment(usetex):
    """
    Test whether mixed math text and normal text are properly aligned along the X-axis.
    Ensures that the vertical bar '||' in "$k$1||\n1||" aligns correctly across lines
    after the fix.
    """
    fig, ax = plt.subplots()

    # Target test text
    txt = "$k$1||\n1||"
    font = FontProperties(size=12)  # Set uniform font size
    text_obj = ax.text(0.5, 0.5, txt, ha="right", va="top", usetex=usetex, fontproperties=font)

    # Trigger layout computation in Matplotlib
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Compute the overall bounding box of the text
    bbox = text_obj.get_window_extent(renderer)

    # Compute the bounding boxes of each line using MathTextParser
    parser = MathTextParser("agg")
    width_math, height_math, *_ = parser.parse("$k$1||", dpi=100, prop=font)
    width_normal, height_normal, *_ = parser.parse("1||", dpi=100, prop=font)

    # Calculate the X-axis misalignment of '||'
    x_offset = abs(width_math - width_normal)

    # If the fix is applied correctly, x_offset should be close to 0
    assert x_offset < 1, f"X misalignment detected: {x_offset}px"
