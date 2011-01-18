
class Container(tuple):
    """
    Base class for containers.
    """

    def __repr__(self):
        return "<Container object of %d artists>" % (len(self))

    def __new__(cls, *kl, **kwargs):
        return tuple.__new__(cls, kl[0])

    def __init__(self, kl, label=None):

        self.eventson = False  # fire events only if eventson
        self._oid = 0  # an observer id
        self._propobservers = {} # a dict from oids to funcs

        self.set_label(label)


    def get_label(self):
        """
        Get the label used for this artist in the legend.
        """
        return self._label

    def set_label(self, s):
        """
        Set the label to *s* for auto legend.

        ACCEPTS: any string
        """
        self._label = s
        self.pchanged()

    def add_callback(self, func):
        """
        Adds a callback function that will be called whenever one of
        the :class:`Artist`'s properties changes.

        Returns an *id* that is useful for removing the callback with
        :meth:`remove_callback` later.
        """
        oid = self._oid
        self._propobservers[oid] = func
        self._oid += 1
        return oid

    def remove_callback(self, oid):
        """
        Remove a callback based on its *id*.

        .. seealso::

            :meth:`add_callback`
               For adding callbacks

        """
        try: del self._propobservers[oid]
        except KeyError: pass

    def pchanged(self):
        """
        Fire an event when property changed, calling all of the
        registered callbacks.
        """
        for oid, func in self._propobservers.items():
            func(self)

    def remove(self):
        for c in self:
            c.remove()


class BarContainer(Container):

    def __init__(self, patches, errorbar=None, **kwargs):
        self.patches = patches
        self.errorbar = errorbar
        Container.__init__(self, patches, **kwargs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.clf()
    bb1 = plt.bar([0, 1, 2], [2, 3, 1], label="test", width=0.4)
    bb2 = plt.bar([0.5, 1.5, 2.5], [2, 3, 1], label="test2", color="red", width=0.4)
    #cont1 = Container(bb1, err=3)


    #aa = BarContainer(("a",), errorbar=1)

