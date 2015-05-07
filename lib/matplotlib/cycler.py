from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from itertools import product
from six.moves import zip, reduce
from operator import mul, add
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

    def _compose(self):
        """
        Compose the 'left' and 'right' components of this cycle
        with the proper operation (zip or product as of now)
        """
        for a, b in self._op(self._left, self._right):
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

    def __getitem__(self, key):
        # TODO : maybe add numpy style fancy slicing
        if isinstance(key, slice):
            trans = self._transpose()
            return reduce(add, (cycler(k, v[key])
                                for k, v in six.iteritems(trans)))
        else:
            raise ValueError("Can only use slices with Cycler.__getitem__")

    def __iter__(self):
        if self._right is None:
            return iter(self._left)

        return self._compose()

    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Can only add equal length cycles, "
                             "not {0} and {1}".format(len(self), len(other)))
        return Cycler(self, other, zip)

    def __mul__(self, other):
        if isinstance(other, Cycler):
            return Cycler(self, other, product)
        elif isinstance(other, int):
            trans = self._transpose()
            return reduce(add, (cycler(k, v*other)
                                for k, v in six.iteritems(trans)))
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __len__(self):
        op_dict = {zip: min, product: mul}
        if self._right is None:
            return len(self._left)
        l_len = len(self._left)
        r_len = len(self._right)
        return op_dict[self._op](l_len, r_len)

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
            itr = list(v[lab] for v in self)
            return "cycler({lab!r}, {itr!r})".format(lab=lab, itr=itr)
        else:
            op = op_map.get(self._op, '?')
            msg = "({left!r} {op} {right!r})"
            return msg.format(left=self._left, op=op, right=self._right)

    def _repr_html_(self):
        # an table showing the value of each key through a full cycle
        output = "<table>"
        for key in self.keys:
            output += "<th>{key!r}</th>".format(key=key)
        for d in iter(self):
            output += "<tr>"
            for val in d.values():
                output += "<td>{val!r}</td>".format(val=val)
            output += "</tr>"
        output += "</table>"
        return output

    def _transpose(self):
        """
        Internal helper function which iterates through the
        styles and returns a dict of lists instead of a list of
        dicts.  This is needed for multiplying by integers and
        for __getitem__

        Returns
        -------
        trans : dict
            dict of lists for the styles
        """

        # TODO : sort out if this is a bottle neck, if there is a better way
        # and if we care.

        keys = self.keys
        out = {k: list() for k in keys}

        for d in self:
            for k in keys:
                out[k].append(d[k])
        return out

    def simplify(self):
        """
        Simplify the Cycler and return as a composition only
        sums (no multiplications)

        Returns
        -------
        simple : Cycler
            An equivalent cycler using only summation
        """
        # TODO: sort out if it is worth the effort to make sure this is
        # balanced.  Currently it is is
        # (((a + b) + c) + d) vs
        # ((a + b) + (c + d))
        # I would believe that there is some performance implications

        trans = self._transpose()
        return reduce(add, (cycler(k, v) for k, v in six.iteritems(trans)))


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
            itr = list(v[lab] for v in itr)

    return Cycler._from_iter(label, itr)
