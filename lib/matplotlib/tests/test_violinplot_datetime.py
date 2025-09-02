"""Tests for datetime and timedelta support in violinplot."""

import datetime
import pytest
import matplotlib.pyplot as plt

def make_vpstats():
    """Create minimal valid stats for a violin plot."""
    datetimes = [
        datetime.datetime(2023, 2, 10),
        datetime.datetime(2023, 5, 18),
        datetime.datetime(2023, 6, 6)
    ]
    return [{
        'coords': datetimes,
        'vals': [0.1, 0.5, 0.2],
        'mean': datetimes[1],
        'median': datetimes[1],
        'min': datetimes[0],
        'max': datetimes[-1],
        'quantiles': datetimes
    }, {
        'coords': datetimes,
        'vals': [0.2, 0.3, 0.4],
        'mean': datetimes[2],
        'median': datetimes[2],
        'min': datetimes[0],
        'max': datetimes[-1],
        'quantiles': datetimes
    }]

def test_datetime_positions_with_float_widths_raises():
    """Test that datetime positions with float widths raise TypeError."""
    fig, ax = plt.subplots()
    try:
        vpstats = make_vpstats()
        positions = [datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)]
        widths = [0.5, 1.0]
        with pytest.raises(TypeError,
        match="positions are datetime/date.*widths as datetime\\.timedelta"):
            ax.violin(vpstats, positions=positions, widths=widths)
    finally:
        plt.close(fig)

def test_datetime_positions_with_scalar_float_width_raises():
    """Test that datetime positions with scalar float width raise TypeError."""
    fig, ax = plt.subplots()
    try:
        vpstats = make_vpstats()
        positions = [datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)]
        widths = 0.75
        with pytest.raises(TypeError,
        match="positions are datetime/date.*widths as datetime\\.timedelta"):
            ax.violin(vpstats, positions=positions, widths=widths)
    finally:
        plt.close(fig)

def test_numeric_positions_with_float_widths_ok():
    """Test that numeric positions with float widths work."""
    fig, ax = plt.subplots()
    try:
        vpstats = make_vpstats()
        positions = [1.0, 2.0]
        widths = [0.5, 1.0]
        ax.violin(vpstats, positions=positions, widths=widths)
    finally:
        plt.close(fig)

def test_mixed_positions_datetime_and_numeric_behaves():
    """Test that mixed datetime and numeric positions with float widths raise TypeError."""
    fig, ax = plt.subplots()
    try:
        vpstats = make_vpstats()
        positions = [datetime.datetime(2020, 1, 1), 2.0]
        widths = [0.5, 1.0]
        with pytest.raises(TypeError,
        match="positions are datetime/date.*widths as datetime\\.timedelta"):
            ax.violin(vpstats, positions=positions, widths=widths)
    finally:
        plt.close(fig)