from matplotlib.cbook import iterable
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
        factors = {('hours', 'seconds'): 3600, ('minutes', 'hours'): 1 / 60,
                   ('minutes', 'seconds'): 60, ('feet', 'miles'): 1 / 5280.,
                   ('feet', 'inches'): 12, ('miles', 'inches'): 12 * 5280}
        if self.units != new_units:
            mult = factors[self.units, new_units]
            return Quantity(mult * self.magnitude, new_units)
        else:
            return Quantity(self.magnitude, self.units)

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)

    def __getitem__(self, item):
        return Quantity(self.magnitude[item], self.units)

    def __array__(self):
        return np.asarray(self.magnitude)


# Tests that the conversion machinery works properly for classes that
# work as a facade over numpy arrays (like pint)
@image_comparison(baseline_images=['plot_pint'],
                  extensions=['png'], remove_text=False, style='mpl20')
def test_numpy_facade():
    # Create an instance of the conversion interface and
    # mock so we can check methods called
    qc = munits.ConversionInterface()

    def convert(value, unit, axis):
        if hasattr(value, 'units'):
            return value.to(unit).magnitude
        elif iterable(value):
            try:
                return [v.to(unit).magnitude for v in value]
            except AttributeError:
                return [Quantity(v, axis.get_units()).to(unit).magnitude
                        for v in value]
        else:
            return Quantity(value, axis.get_units()).to(unit).magnitude

    qc.convert = MagicMock(side_effect=convert)
    qc.axisinfo = MagicMock(side_effect=lambda u, a: munits.AxisInfo(label=u))
    qc.default_units = MagicMock(side_effect=lambda x, a: x.units)

    # Register the class
    munits.registry[Quantity] = qc

    # Simple test
    y = Quantity(np.linspace(0, 30), 'miles')
    x = Quantity(np.linspace(0, 5), 'hours')

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15)  # Make space for label
    ax.plot(x, y, 'tab:blue')
    ax.axhline(Quantity(26400, 'feet'), color='tab:red')
    ax.axvline(Quantity(120, 'minutes'), color='tab:green')
    ax.yaxis.set_units('inches')
    ax.xaxis.set_units('seconds')

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
