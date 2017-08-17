import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.units as munits
import numpy as np

try:
    # mock in python 3.3+
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


# Basic class that wraps numpy array and has units
class Quantity(object):
    def __init__(self, data, units):
        self.magnitude = data
        self.units = units

    def to(self, new_units):
        return Quantity(self.magnitude, new_units)

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)

    def __getitem__(self, item):
        return self.magnitude[item]


# Tests that the conversion machinery works properly for classes that
# work as a facade over numpy arrays (like pint)
def test_numpy_facade():
    # Create an instance of the conversion interface and
    # mock so we can check methods called
    qc = munits.ConversionInterface()

    def convert(value, unit, axis):
        if hasattr(value, 'units'):
            return value.to(unit)
        else:
            return Quantity(value, axis.get_units()).to(unit).magnitude

    qc.convert = MagicMock(side_effect=convert)
    qc.axisinfo = MagicMock(return_value=None)
    qc.default_units = MagicMock(side_effect=lambda x, a: x.units)

    # Register the class
    munits.registry[Quantity] = qc

    # Simple test
    t = Quantity(np.linspace(0, 10), 'sec')
    d = Quantity(30 * np.linspace(0, 10), 'm/s')

    fig, ax = plt.subplots(1, 1)
    l, = plt.plot(t, d)
    ax.yaxis.set_units('inch')

    assert qc.convert.called
    assert qc.axisinfo.called
    assert qc.default_units.called


# Tests gh-8908
@image_comparison(baseline_images=['plot_masked_units'],
                  extensions=['png'], remove_text=True, style='mpl20')
def test_plot_masked_units():
    data = np.linspace(-5, 5)
    data_masked = np.ma.array(data, mask=(data > -2) & (data < 2))
    data_masked_units = Quantity(data_masked, 'meters')

    fig, ax = plt.subplots()
    ax.plot(data_masked_units)
