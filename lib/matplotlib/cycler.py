from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from itertools import product, cycle
from six.moves import zip
import copy


def _process_keys(left, right):
    """
    Helper function to compose cycler keys

    Parameters
    ----------
    left, right : Cycler or None
        The cyclers to be composed
    Returns
    -------
    keys : set
        The keys in the composition of the two cyclers
    """
    l_key = left.keys if left is not None else set()
    r_key = right.keys if right is not None else set()
    if l_key & r_key:
        raise ValueError("Can not compose overlapping cycles")
    return l_key | r_key


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
        self._keys = _process_keys(left, right)
        self._left = copy.copy(left)
        self._right = copy.copy(right)
        self._op = op

    @property
    def keys(self):
        return set(self._keys)

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

    def to_list(self):
        """
        Return a list of the dictionaries yielded by
        this Cycler.

        Returns
        -------
        cycle : list
            All of the dictionaries yielded by this Cycler in order.
        """
        return list(self.finite_iter())

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
    def _from_iter(cls, label, itr):
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

    def __iadd__(self, other):
        old_self = copy.copy(self)
        self._keys = _process_keys(old_self, other)
        self._left = old_self
        self._op = zip
        self._right = copy.copy(other)
        return self

    def __imul__(self, other):
        old_self = copy.copy(self)
        self._keys = _process_keys(old_self, other)
        self._left = old_self
        self._op = product
        self._right = copy.copy(other)
        return self

    def __repr__(self):
        op_map = {zip: '+', product: '*'}
        if self._right is None:
            lab = self.keys.pop()
            itr = list(v[lab] for v in self.finite_iter())
            return "cycler({lab!r}, {itr!r})".format(lab=lab, itr=itr)
        else:
            op = op_map.get(self._op, '?')
            msg = "({left!r} {op} {right!r})"
            return msg.format(left=self._left, op=op, right=self._right)


def cycler(label, itr):
    """
    Create a new `Cycler` object from a property name and
    iterable of values.

    Parameters
    ----------
    label : str
        The property key.

    itr : iterable
        Finite length iterable of the property values.

    Returns
    -------
    cycler : Cycler
        New `Cycler` for the given property
    """
    if isinstance(itr, Cycler):
        keys = itr.keys
        if len(keys) != 1:
            msg = "Can not create Cycler from a multi-property Cycler"
            raise ValueError(msg)

        if label in keys:
            return copy.copy(itr)
        else:
            lab = keys.pop()
            itr = list(v[lab] for v in itr.finite_iter())

    return Cycler._from_iter(label, itr)
