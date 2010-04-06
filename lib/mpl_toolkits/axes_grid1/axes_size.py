
"""
provides a classese of simlpe units that will be used with AxesDivider
class (or others) to determine the size of each axes. The unit
classes define `get_size` method that returns a tuple of two floats,
meaning relative and absolute sizes, respectively.

Note that this class is nothing more than a simple tuple of two
floats. Take a look at the Divider class to see how these two
values are used.

"""

import matplotlib.cbook as cbook


class _Base(object):
    "Base class"
    pass

class Fixed(_Base):
    "Simple fixed size  with absolute part = *fixed_size* and relative part = 0"
    def __init__(self, fixed_size):
        self.fixed_size = fixed_size

    def get_size(self, renderer):
        rel_size = 0.
        abs_size = self.fixed_size
        return rel_size, abs_size


class Scaled(_Base):
    "Simple scaled(?) size with absolute part = 0 and relative part = *scalable_size*"
    def __init__(self, scalable_size):
        self._scalable_size = scalable_size

    def get_size(self, renderer):
        rel_size = self._scalable_size
        abs_size = 0.
        return rel_size, abs_size

Scalable=Scaled


class AxesX(_Base):
    """
    Scaled size whose relative part corresponds to the data width
    of the *axes* multiplied by the *aspect*.
    """
    def __init__(self, axes, aspect=1.):
        self._axes = axes
        self._aspect = aspect

    def get_size(self, renderer):
        l1, l2 = self._axes.get_xlim()
        rel_size = abs(l2-l1)*self._aspect
        abs_size = 0.
        return rel_size, abs_size

class AxesY(_Base):
    """
    Scaled size whose relative part corresponds to the data height
    of the *axes* multiplied by the *aspect*.
    """
    def __init__(self, axes, aspect=1.):
        self._axes = axes
        self._aspect = aspect

    def get_size(self, renderer):
        l1, l2 = self._axes.get_ylim()
        rel_size = abs(l2-l1)*self._aspect
        abs_size = 0.
        return rel_size, abs_size


class MaxExtent(_Base):
    """
    Size whose absolute part is the largest width (or height) of
    the given *artist_list*.
    """
    def __init__(self, artist_list, w_or_h):
        self._artist_list = artist_list

        if w_or_h not in ["width", "height"]:
            raise ValueError()

        self._w_or_h = w_or_h

    def add_artist(self, a):
        self._artist_list.append(a)

    def get_size(self, renderer):
        rel_size = 0.
        w_list, h_list = [], []
        for a in self._artist_list:
            bb = a.get_window_extent(renderer)
            w_list.append(bb.width)
            h_list.append(bb.height)
        dpi = a.get_figure().get_dpi()
        if self._w_or_h == "width":
            abs_size = max(w_list)/dpi
        elif self._w_or_h == "height":
            abs_size = max(h_list)/dpi

        return rel_size, abs_size


class MaxWidth(_Base):
    """
    Size whose absolute part is the largest width of
    the given *artist_list*.
    """
    def __init__(self, artist_list):
        self._artist_list = artist_list

    def add_artist(self, a):
        self._artist_list.append(a)

    def get_size(self, renderer):
        rel_size = 0.
        w_list = []
        for a in self._artist_list:
            bb = a.get_window_extent(renderer)
            w_list.append(bb.width)
        dpi = a.get_figure().get_dpi()
        abs_size = max(w_list)/dpi

        return rel_size, abs_size



class MaxHeight(_Base):
    """
    Size whose absolute part is the largest height of
    the given *artist_list*.
    """
    def __init__(self, artist_list):
        self._artist_list = artist_list

    def add_artist(self, a):
        self._artist_list.append(a)

    def get_size(self, renderer):
        rel_size = 0.
        h_list = []
        for a in self._artist_list:
            bb = a.get_window_extent(renderer)
            h_list.append(bb.height)
        dpi = a.get_figure().get_dpi()
        abs_size = max(h_list)/dpi

        return rel_size, abs_size


class Fraction(_Base):
    """
    An instance whose size is a *fraction* of the *ref_size*.

      >>> s = Fraction(0.3, AxesX(ax))

    """
    def __init__(self, fraction, ref_size):
        self._fraction_ref = ref_size
        self._fraction = fraction

    def get_size(self, renderer):
        if self._fraction_ref is None:
            return self._fraction, 0.
        else:
            r, a = self._fraction_ref.get_size(renderer)
            rel_size = r*self._fraction
            abs_size = a*self._fraction
            return rel_size, abs_size

class Padded(_Base):
    """
    Return a instance where the absolute part of *size* is
    increase by the amount of *pad*.
    """
    def __init__(self, size, pad):
        self._size = size
        self._pad = pad

    def get_size(self, renderer):
        r, a = self._size.get_size(renderer)
        rel_size = r
        abs_size = a + self._pad
        return rel_size, abs_size

def from_any(size, fraction_ref=None):
    """
    Creates Fixed unit when the first argument is a float, or a
    Fraction unit if that is a string that ends with %. The second
    argument is only meaningful when Fraction unit is created.

      >>> a = Size.from_any(1.2) # => Size.Fixed(1.2)
      >>> Size.from_any("50%", a) # => Size.Fraction(0.5, a)

    """
    if cbook.is_numlike(size):
        return Fixed(size)
    elif cbook.is_string_like(size):
        if size[-1] == "%":
            return Fraction(float(size[:-1])/100., fraction_ref)

    raise ValueError("Unknown format")


