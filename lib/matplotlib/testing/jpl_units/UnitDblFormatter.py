#===========================================================================
#
# UnitDblFormatter
#
#===========================================================================


"""UnitDblFormatter module containing class UnitDblFormatter."""

#===========================================================================
# Place all imports after here.
#
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import matplotlib.ticker as ticker
#
# Place all imports before here.
#===========================================================================

__all__ = [ 'UnitDblFormatter' ]

#===========================================================================
class UnitDblFormatter(ticker.Formatter):
    """The formatter for UnitDbl data types.  This allows for formatting
       with the unit string.
    """
    def format_for_tick(self, value, pos=None):
        return str(value)
