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
    return Cycler._from_iter(label, itr)


class BaseCycler(object):
    """
    Helper base-class to provide composition logic to
    `SingleCycler` and `CompoundCycler`.

    This class
    does not have a `__init__` method which will result
    in a usable object.
    """
    def __iter__(self):
        return cycle(self.finite_iter())

    def __add__(self, other):
        return CompoundCycler(self, other, zip)

    def __mul__(self, other):
        return CompoundCycler(self, other, product)


class SingleCycler(BaseCycler):
    """
    Class to hold the cycle for a single parameter and handle the
    composition.

    Parameters
    ----------
    label : str
        The name of the property this cycles over
    itr : iterable
        Finite length iterable of the property values.
    """
    def __init__(self, label, itr):
        self._itr = itr
        self._label = label

    def finite_iter(self):
        """
        Return a finite iterator over the configurations in
        this cycle.

        Returns
        -------
        gen : generator
            A generator that yields dictionaries keyed on the property
            name of the values to be used

        """
        return ({self._label: v} for v in self._itr)

    def __len__(self):
        return len(self._itr)

    @property
    def keys(self):
        """
        The properties that this cycle loops over
        """
        return set([self._label])


class CompoundCycler(BaseCycler):
    """
    A class to handle cycling multiple artist properties.

    This class has two compositions methods '+' for 'inner'
    products of the cycles and '*' for outer products of the
    cycles.

    This objects should not be created directly, but instead
    result of composition of existing `SingleCycler` and
    `CompoundCycler` objects.

    Parameters
    ----------
    left : BaseCycler
        The 'left' cycler

    right : BaseCycler
        The 'right' cycler

    op : function
        Function which composes the 'left' and 'right' cyclers.

    """
    def __init__(self, left, right, op):
        self._left = left
        self._right = right
        self._op = op
        l_key = left.keys
        r_key = right.keys
        if l_key & r_key:
            raise ValueError("Can not compose overlapping cycles")
        self._keys = l_key | r_key

    def finite_iter(self):
        """
        Return a finite iterator over the configurations in
        this cycle.

        Returns
        -------
        gen : generator
            A generator that yields dictionaries keyed on the property
            name of the values to be used
        """
        return self._compose()

    def _compose(self):
        """
        Private function to handle the logic of merging the dictionaries
        of the left and right cycles.

        Yields
        ------
        ret : dict
            A dictionary keyed on the property name of the values to be used
        """
        for a, b in self._op(self._left.finite_iter(),
                             self._right.finite_iter()):
            out = dict()
            out.update(a)
            out.update(b)
            yield out

    def __len__(self):
        return len(list(self.finite_iter()))

    @property
    def keys(self):
        """
        The properties that this cycle loops over
        """
        return self._keys
