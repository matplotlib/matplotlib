import matplotlib.pyplot as plt
import pytest

def test_subfigure_mosaic_basic():
    fig, subfigs = plt.subplot_mosaic([
        ['A', 'B'],
        ['C', 'D']
    ], constrained_layout=True)
    
    assert 'A' in subfigs
    assert 'B' in subfigs
    assert 'C' in subfigs
    assert 'D' in subfigs
    assert len(subfigs) == 4

def test_subfigure_mosaic_nested():
    fig, subfigs = plt.subplot_mosaic([
        ['A', 'B1'],
        ['A', 'B2'],
        ['C', 'D']
    ], constrained_layout=True)
    
    assert 'A' in subfigs
    assert 'B1' in subfigs
    assert 'B2' in subfigs
    assert 'C' in subfigs
    assert 'D' in subfigs
    assert len(subfigs) == 5

def test_subfigure_mosaic_empty_sentinel():
    fig, subfigs = plt.subplot_mosaic([
        ['A', '.'],
        ['C', 'D']
    ], empty_sentinel='.', constrained_layout=True)
    
    assert 'A' in subfigs
    assert 'C' in subfigs
    assert 'D' in subfigs
    assert '.' not in subfigs
    assert len(subfigs) == 3
