from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from itertools import product, cycle
from six.moves import zip


class Cycler(object):
    """
    A class to handle cycling multiple artist properties.

    This class has two compositions methods '+' for 'inner'
    products of the cycles and '*' for outer products of the
    cycles.

    Parameters
    ----------
    left : Cycler or None
        The 'left' cycler

    right : Cycler or None
        The 'right' cycler

    op : func or None
        Function which composes the 'left' and 'right' cyclers.

    """
    def __init__(self, left, right=None, op=None):
        self._left = left
        self._right = right
        self._op = op
        l_key = left.keys if left is not None else set()
        r_key = right.keys if right is not None else set()
        if l_key & r_key:
            raise ValueError("Can not compose overlapping cycles")
        self._keys = l_key | r_key

    @property
    def keys(self):
        return self._keys

    def finite_iter(self):
        """
        Return a finite iterator over the configurations in
        this cycle.
        """
        if self._right is None:
            try:
                return self._left.finite_iter()
            except AttributeError:
                return iter(self._left)
        return self._compose()

    def _compose(self):
        """
        Compose the 'left' and 'right' components of this cycle
        with the proper operation (zip or product as of now)
        """
        for a, b in self._op(self._left.finite_iter(),
                             self._right.finite_iter()):
            out = dict()
            out.update(a)
            out.update(b)
            yield out

    @classmethod
    def from_iter(cls, label, itr):
        """
        Class method to create 'base' Cycler objects
        that do not have a 'right' or 'op' and for which
        the 'left' object is not another Cycler.

        Parameters
        ----------
        label : str
            The property key.

        itr : iterable
            Finite length iterable of the property values.

        Returns
        -------
        cycler : Cycler
            New 'base' `Cycler`
        """
        ret = cls(None)
        ret._left = list({label: v} for v in itr)
        ret._keys = set([label])
        return ret

    def __iter__(self):
        return cycle(self.finite_iter())

    def __add__(self, other):
        return Cycler(self, other, zip)

    def __mul__(self, other):
        return Cycler(self, other, product)

    def __len__(self):
        return len(list(self.finite_iter()))
