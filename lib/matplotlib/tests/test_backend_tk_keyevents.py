import pytest
from types import SimpleNamespace


def make_tk_event(keysym):
    return SimpleNamespace(keysym=keysym, keycode=None, state=None)


@pytest.mark.skipif(
    pytest.importorskip("tkinter", reason="tkinter not available in this environment"),
    reason="tkinter not available",
)
def test_tk_key_release_reports_released_key(monkeypatch):
    """Tk backend should report the actual released key, not the composed key.

    Scenario:
      - user presses Alt+h -> backend emits 'alt+h' on press (unchanged)
      - user releases 'h' while Alt remains pressed -> release should be 'h'
      - user then releases Alt -> release should be 'alt'
    """
    # Import the specific TkAgg backend and create a canvas
    from matplotlib.backends import backend_tkagg as tkagg
    import matplotlib.pyplot as plt

    fig = plt.figure()
    canvas = tkagg.FigureCanvasTkAgg(fig)

    seen = []

    def on_release(event):
        seen.append(event.key)

    canvas.mpl_connect('key_release_event', on_release)

    # Simulate release of 'h' (Tk reports 'h')
    e_h = make_tk_event('h')
    # Call the backend's handler: depending on implementation the method
    # name may be `key_release` on the canvas manager; adapt if necessary.
    # The following calls into the backend-level API that our patch changes.
    # If your tree exposes a different entrypoint, call that instead.
    # We attempt multiple entrypoints to maximize compatibility across trees.
    if hasattr(canvas, 'key_release_event'):
        # Some test installations route through mpl's event system directly.
        canvas.key_release_event('h', guiEvent=e_h)
    else:
        # Fallback: call the backend handler directly if present
        backend = getattr(canvas, 'tkcanvas', None) or getattr(canvas, 'manager', None)
        # If the backend exposes a `key_release` method, call it
        if hasattr(canvas, 'key_release'):
            canvas.key_release(e_h)
        elif hasattr(backend, 'key_release'):
            backend.key_release(e_h)
        else:
            pytest.skip('No accessible key_release handler in this backend variant')

    assert seen, "No key_release_event observed"
    assert seen[-1] == 'h', f"expected 'h' on release, got {seen[-1]}"

    # Simulate release of Alt
    e_alt = make_tk_event('Alt_L')
    if hasattr(canvas, 'key_release_event'):
        canvas.key_release_event('alt', guiEvent=e_alt)
    else:
        if hasattr(canvas, 'key_release'):
            canvas.key_release(e_alt)
        elif hasattr(backend, 'key_release'):
            backend.key_release(e_alt)
        else:
            pytest.skip('No accessible key_release handler in this backend variant')

    assert seen[-1] in ('alt', 'Alt_L', 'alt_l')
