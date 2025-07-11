import sys
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.font_manager import FontSuperfamily
from matplotlib.font_manager import FontProperties, _default_superfamilies


# Define a testable superfamily registry for isolated tests
def setup_superfamily():
    sf = FontSuperfamily.get_superfamily("TestFamily")
    sf.register("serif", "Test Serif")
    sf.register("serif", "Test Serif Bold", weight="bold")
    sf.register("sans", "Test Sans")
    sf.register("mono", "Test Mono", weight="bold", style="italic")
    return sf


# Create test functions
def test_register_and_get_family_default():
    sf = setup_superfamily()
    assert sf.get_family("serif") == "Test Serif"


def test_get_family_with_weight():
    sf = setup_superfamily()
    assert sf.get_family("serif", weight="bold") == "Test Serif Bold"


def test_get_family_with_style_and_weight():
    sf = setup_superfamily()
    assert sf.get_family("mono", weight="bold", style="italic") == "Test Mono"


def test_get_family_fallback_to_default():
    sf = setup_superfamily()
    # This should fallback to "normal-normal" entry
    assert sf.get_family("sans", weight="light") == "Test Sans"


def test_get_family_not_found_returns_none():
    sf = setup_superfamily()
    assert sf.get_family("fantasy") is None


def test_superfamily_takes_precedence_over_genre_family(monkeypatch):
    """
    If both font.superfamily and font.serif are defined,
    font.superfamily should take precedence.
    """
    setup_superfamily()
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "TestFamily")
    monkeypatch.setitem(mpl.rcParams, "font.family", ["serif"])
    monkeypatch.setitem(mpl.rcParams, "font.serif", ["Some Custom Serif Font"])

    fp = FontProperties(weight="bold")
    # Should resolve to 'Test Serif Bold', not 'Some Custom Serif Font'
    assert fp.get_family() == ["Test Serif Bold"]


def test_fontproperties_with_superfamily(monkeypatch):
    """Validate FontProperties resolves the superfamily correctly"""
    setup_superfamily()
    # Inject rcParams temporarily
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "TestFamily")
    monkeypatch.setitem(mpl.rcParams, "font.family", "serif")

    fp = FontProperties(weight="bold")
    assert fp.get_family() == ["Test Serif Bold"]


def test_superfamily_can_be_disabled(monkeypatch):
    """
    Ensure that font.superfamily can be disabled using None or 'None'.
    """
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", None)
    monkeypatch.setitem(mpl.rcParams, "font.family", ["serif"])

    fp = FontProperties()
    assert fp.get_family() == ["serif"]

    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "None")
    fp2 = FontProperties()
    assert fp2.get_family() == ["serif"]


def test_fontproperties_without_superfamily(monkeypatch):
    monkeypatch.setitem(mpl.rcParams, "font.family", "serif")
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", None)

    fp = FontProperties()
    # Should not use the superfamily logic, and preserve original family
    assert fp.get_family() == ["serif"]


def test_get_family_with_nonexistent_weight_style_combination():
    sf = setup_superfamily()
    # Should fall back to default genre if exact match for weight+style doesn't exist
    assert sf.get_family("mono", weight="bold", style="oblique") == "Test Mono"


def test_fontproperties_superfamily_partial_match(monkeypatch):
    # Only genre match exists, weight and style do not
    setup_superfamily()
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "TestFamily")
    monkeypatch.setitem(mpl.rcParams, "font.family", "sans")

    # No specific weight/style for sans, should still resolve
    fp = FontProperties(weight="black", style="italic")
    assert fp.get_family() == ["Test Sans"]


def test_fontproperties_superfamily_not_defined(monkeypatch):
    # Superfamily name exists but no mapping for genre
    setup_superfamily()
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "TestFamily")
    monkeypatch.setitem(mpl.rcParams, "font.family", "fantasy")

    fp = FontProperties()
    # Should fall back to original family
    assert fp.get_family() == ["fantasy"]


def test_fontproperties_superfamily_unknown(monkeypatch):
    # Non-existent superfamily
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "UnknownFamily")
    monkeypatch.setitem(mpl.rcParams, "font.family", "serif")

    fp = FontProperties()
    # Should fall back to family as superfamily doesn't exist
    assert fp.get_family() == ["serif"]


def test_superfamily_render_with_dejavu(monkeypatch):
    """
    Ensure that the default registered superfamily 'DejaVu'
    resolves and applies 'DejaVu Sans'
    at render time using FontProperties and rcParams.
    """

    # Guarantee the font is in the known system fonts
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    assert "DejaVu Sans Mono" in available_fonts, "DejaVu Sans must be available"

    # Ensure default superfamilies are loaded
    FontSuperfamily._registry.clear()
    FontSuperfamily._populate_default_superfamilies()

    # Set rcParams to trigger superfamily logic
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "DejaVu")
    monkeypatch.setitem(mpl.rcParams, "font.family", ["mono"])
    monkeypatch.setitem(mpl.rcParams, "font.style", "normal")
    monkeypatch.setitem(mpl.rcParams, "font.weight", "normal")

    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    text = ax.text(0.5, 0.5, "DejaVu test", ha="center")

    # Trigger draw to resolve font fully
    canvas.draw()

    actual_font = text.get_fontproperties().get_name()
    assert actual_font == "DejaVu Sans Mono"


def test_superfamily_overrides_genre_specific_list(monkeypatch):
    """
    Verify that font.superfamily takes precedence over font.family and
    genre-specific fallback lists like font.serif during rendering.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib import font_manager

    # Ensure default superfamilies are available
    FontSuperfamily._registry.clear()
    FontSuperfamily._populate_default_superfamilies()

    # Use a known default superfamily
    superfamily = "DejaVu"
    expected_font = "DejaVu Serif"  # Must exist in the default registry

    # Ensure the font is present in the system
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if expected_font not in available_fonts:
        pytest.skip(f"{expected_font} is not available")

    # Apply conflicting font settings
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", superfamily)
    monkeypatch.setitem(mpl.rcParams, "font.family", ["serif"])
    monkeypatch.setitem(mpl.rcParams, "font.serif",
                        ["Times New Roman"])  # Should be ignored

    # Plot and trigger font resolution
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    text_obj = ax.text(0.5, 0.5, "Testing precedence", ha="center")
    canvas.draw()

    # Get actual font used
    actual_font = text_obj.get_fontproperties().get_name()

    assert actual_font == expected_font, (
        f"Expected font '{expected_font}' from superfamily '{superfamily}', "
        f"but got '{actual_font}'"
    )


@pytest.mark.parametrize("sf_data", _default_superfamilies)
def test_registered_superfamilies_resolve_if_fonts_exist(sf_data, monkeypatch):
    """
    Verify that each registered superfamily correctly resolves to its expected
    default 'normal-normal' font per genre, provided the font is installed.
    """

    sf_name = sf_data["name"]
    expected_fonts = []

    # Collect (genre, expected_font) for each "normal-normal" mapping
    for genre, variants in sf_data["variants"].items():
        if "normal-normal" in variants:
            expected_fonts.append((genre, variants["normal-normal"]))

    # Skip test if none of the fonts are available in the system
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if all(expected not in available_fonts for _, expected in expected_fonts):
        pytest.skip(f"None of the fonts for superfamily '{sf_name}' are available")

    # Apply the superfamily setting in rcParams
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", sf_name)

    for genre, expected_font in expected_fonts:
        # Set genre in rcParams
        genre_rc = {
            "sans": "sans-serif",
            "serif": "serif",
            "mono": "monospace",
            "cursive": "cursive",
            "fantasy": "fantasy"
        }.get(genre, genre)

        monkeypatch.setitem(mpl.rcParams, "font.family", [genre_rc])

        # Use FontProperties to trigger resolution
        fp = FontProperties()
        resolved = fp.get_family()

        assert isinstance(resolved, list)
        assert expected_font in resolved, (
            f"Exp'{expected_font}' from sf '{sf_name}' with genre '{genre}', "
            f"but got {resolved}"
        )


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="This test only runs on macOS")
def test_superfamily_render_with_apple_fonts(monkeypatch):
    """
    Test that a superfamily composed of macOS default fonts:
    Helvetica, Times New Roman, Courier New;
    correctly resolves and is used for rendering
    via FontProperties and rcParams.
    """
    # Ensure expected system fonts are available
    required_fonts = {"Helvetica", "Times New Roman", "Courier New"}
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if not required_fonts.issubset(available_fonts):
        pytest.skip("macOS system fonts not all available")

    # Define and register a superfamily using macOS default fonts

    sf = FontSuperfamily.get_superfamily("Apple")
    sf.register("serif", "Times New Roman")
    sf.register("sans", "Helvetica")
    sf.register("mono", "Courier New")

    # Trigger superfamily resolution via rcParams
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", "Apple")
    monkeypatch.setitem(mpl.rcParams, "font.family", ["sans"])
    #monkeypatch.setitem(mpl.rcParams, "font.style", "normal")
    #monkeypatch.setitem(mpl.rcParams, "font.weight", "normal")

    # Create and render a figure
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    text = ax.text(0.5, 0.5, "Test Apple Font", ha="center")
    canvas.draw()

    # Check the resolved font name
    actual_font = text.get_fontproperties().get_name()
    assert actual_font == "Helvetica"


def test_font_superfamily_dict_registration_and_resolution(monkeypatch):
    """
    Ensure that a dict-based font.superfamily entry is correctly registered
    and resolves the expected font name through FontProperties.
    """
    # Clear registry to isolate test
    FontSuperfamily._registry.clear()

    # Define a new superfamily directly via dict
    test_superfamily_dict = {
        "name": "CustomFamily",
        "variants": {
            "sans": {
                "normal-normal": "Arial"
            },
            "serif": {
                "normal-normal": "Times New Roman"
            }
        }
    }

    # Set rcParams using the dict
    monkeypatch.setitem(mpl.rcParams, "font.superfamily", test_superfamily_dict)
    monkeypatch.setitem(mpl.rcParams, "font.family", ["serif"])
    monkeypatch.setitem(mpl.rcParams, "font.style", "normal")
    monkeypatch.setitem(mpl.rcParams, "font.weight", "normal")

    fp = FontProperties()

    # It should now resolve to the font registered in the dict
    resolved = fp.get_family()
    assert isinstance(resolved, list)
    assert "Times New Roman" in resolved

    # Registry should contain the new superfamily under the right name
    assert "CustomFamily" in FontSuperfamily._registry
    register =FontSuperfamily._registry["CustomFamily"].get_family("serif")
    assert register == "Times New Roman"
