import matplotlib
from matplotlib.cbook import iterable, flatten


class ConversionInterface:

    def tickers(x, unit):
        'return (majorloc, minorloc, majorfmt, minorfmt) or None to accept defaults'
        return None
    tickers = staticmethod(tickers)

    def convert_to_value(obj, unit):
        """
        convert obj using unit.  If obj is a sequence, return the
        converted sequence
        """
        return obj
    convert_to_value = staticmethod(convert_to_value)
    
    
class UnitsManager:
    """
    manage unit conversion.

    attribute converters is a dict mapping object class-> conversion interface
    """
    def __init__(self):
        self.converters = {}
        self._cached = {}

    def get_converter(self, x):
        'get the converter interface for x, if any'

        idx = id(x)
        cached = self._cached.get(idx)
        if cached is not None: return cached

        converter = None
        classx = getattr(x, '__class__', None)
                
        if classx is not None:            
            converter = self.converters.get(classx)

        if converter is None and iterable(x):
            for thisx in x:
                classx = getattr(thisx, '__class__', None)
                break
            if classx is not None:            
                converter = self.converters.get(classx)

        if converter is not None:
            self._cached[idx] = converter
        return converter
        
        
    def convert(self, x, unit):
        converter = self.get_converter(x)
        if converter is not None:
            return converter.convert_to_value(x, unit)
        return x

    def tickers(self, x, unit):
        converter = self.get_converter(x)
        if converter is not None:
            return converter.tickers(x, unit)
        return None

class DonothingManager:

    def __init__(self):
        self.converters = {}

    def get_converter(self, x):
        'get the converter interface for x, if any'
        return None
        
    def convert(self, x, unit):
        return x

    def tickers(self, x, unit):
        return None


if matplotlib.rcParams['units']:
    manager = UnitsManager()
else:
    manager = DonothingManager()
    