
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from matplotlib.overlay_manager import LayerSpec
import numpy as np
import os

def test_overlay_visual_parity(tmp_path):
    """
    Verify that plt.overlay produces an identical image to a manually constructed plot
    with twinx, zorder manual setting, and unified legend.
    """
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x) * 10
    
    # --- Manual Implementation "Legacy" ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # Bar layer (primary) - zorder 1
    ax1.bar(x[::10], y1[::10], width=0.5, label='Bar', color='blue', zorder=1)
    
    # Line layer (secondary) - zorder 2
    ax2 = ax1.twinx()
    ax2.plot(x, y2, label='Line', color='red', zorder=2)
    
    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    
    path1 = tmp_path / "manual_plot.png"
    fig1.savefig(path1)
    plt.close(fig1)
    
    # --- Overlay Implementation ---
    fig2, ax_overlay = plt.subplots(figsize=(6, 4))
    
    layers = [
        LayerSpec(method='bar', data=(x[::10], y1[::10]), kwargs={'width': 0.5, 'label': 'Bar', 'color': 'blue'}, axis='primary'),
        LayerSpec(method='plot', data=(x, y2), kwargs={'label': 'Line', 'color': 'red'}, axis='secondary'),
    ]
    
    plt.overlay(layers, ax=ax_overlay)
    
    path2 = tmp_path / "overlay_plot.png"
    fig2.savefig(path2)
    plt.close(fig2)
    
    # Strict comparison
    result = compare_images(str(path1), str(path2), tol=0.0)
    assert result is None, f"Images differ: {result}"

def test_zorder_defaults():
    """Verify default z-orders are applied correctly."""
    fig, ax = plt.subplots()
    layers = [
        LayerSpec('bar', ([0,1], [1,2])),      # Should be 1
        LayerSpec('plot', ([0,1], [3,4])),     # Should be 2
        LayerSpec('scatter', ([0,1], [5,6])),  # Should be 3
    ]
    
    coord = plt.overlay(layers, ax=ax)
    
    # Inspect children of the axis
    # bar creates patches
    bars = [c for c in ax.patches if 'Rectangle' in str(type(c))]
    # plot creates lines
    lines = ax.lines
    # scatter creates collections
    collections = ax.collections
    
    # Check zorders
    # logic: coordinate adds layers.
    # We need to find the specific artists.
    # Simplify: check the last added items of each type
    
    # This might be tricky if defaults overlap or if matplotlib changes internals.
    # But based on our code:
    assert ax.patches[0].get_zorder() == 1
    assert ax.lines[0].get_zorder() == 2
    assert ax.collections[0].get_zorder() == 3
    
    plt.close(fig)

def test_state_integrity():
    """Ensure plt.gca() is the primary axis after finalize."""
    fig = plt.figure()
    ax_primary = fig.add_subplot(111)
    
    layers = [
        LayerSpec('plot', ([0,1], [0,1]), axis='primary'),
        LayerSpec('plot', ([0,1], [0,1]), axis='secondary')
    ]
    
    plt.overlay(layers, ax=ax_primary)
    
    assert plt.gca() is ax_primary
    plt.close(fig)

def test_legend_unification():
    """Verify legend contains handles from both axes."""
    fig, ax = plt.subplots()
    layers = [
        LayerSpec('plot', ([0], [0]), kwargs={'label': 'L1'}, axis='primary'),
        LayerSpec('plot', ([0], [0]), kwargs={'label': 'L2'}, axis='secondary'),
    ]
    
    plt.overlay(layers, ax=ax)
    
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert 'L1' in texts
    assert 'L2' in texts
    
    plt.close(fig)
