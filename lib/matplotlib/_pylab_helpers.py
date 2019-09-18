"""
Manage figures for the pyplot interface.
"""

import atexit
from collections import OrderedDict
import gc


class Gcf:
    """
    Singleton to maintain the relation between figures and their managers, and
    keep track of and "active" figure and manager.

    The canvas of a figure created through pyplot is associated with a figure
    manager, which handles the interaction between the figure and the backend.
    pyplot keeps track of figure managers using an identifier, the "figure
    number" or "manager number" (which can actually be any hashable value);
    this number is available as the :attr:`number` attribute of the manager.

    This class is never instantiated; it consists of an `OrderedDict` mapping
    figure/manager numbers to managers, and a set of class methods that
    manipulate this `OrderedDict`.

    Attributes
    ----------
    figs : OrderedDict
        `OrderedDict` mapping numbers to managers; the active manager is at the
        end.
    """

    figs = OrderedDict()

    @classmethod
    def get_fig_manager(cls, num):
        """
        If manager number *num* exists, make it the active one and return it;
        otherwise return *None*.
        """
        manager = cls.figs.get(num, None)
        if manager is not None:
            cls.set_active(manager)
        return manager

    @classmethod
    def destroy(cls, num):
        """
        Destroy figure number *num*.

        In the interactive backends, this is bound to the window "destroy" and
        "delete" events.
        """
        if not cls.has_fignum(num):
            return
        manager = cls.figs.pop(num)
        manager.canvas.mpl_disconnect(manager._cidgcf)
        manager.destroy()
        gc.collect(1)

    @classmethod
    def destroy_fig(cls, fig):
        """Destroy figure *fig*."""
        canvas = getattr(fig, "canvas", None)
        manager = getattr(canvas, "manager", None)
        num = getattr(manager, "num", None)
        cls.destroy(num)

    @classmethod
    def destroy_all(cls):
        """Destroy all figures."""
        # Reimport gc in case the module globals have already been removed
        # during interpreter shutdown.
        import gc
        for manager in list(cls.figs.values()):
            manager.canvas.mpl_disconnect(manager._cidgcf)
            manager.destroy()
        cls.figs.clear()
        gc.collect(1)

    @classmethod
    def has_fignum(cls, num):
        """Return whether figure number *num* exists."""
        return num in cls.figs

    @classmethod
    def get_all_fig_managers(cls):
        """Return a list of figure managers."""
        return list(cls.figs.values())

    @classmethod
    def get_num_fig_managers(cls):
        """Return the number of figures being managed."""
        return len(cls.figs)

    @classmethod
    def get_active(cls):
        """Return the active manager, or *None* if there is no manager."""
        return next(reversed(cls.figs.values())) if cls.figs else None

    @classmethod
    def set_active(cls, manager):
        """Make *manager* the active manager."""
        cls.figs[manager.num] = manager
        cls.figs.move_to_end(manager.num)

    @classmethod
    def draw_all(cls, force=False):
        """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """
        for manager in cls.get_all_fig_managers():
            if force or manager.canvas.figure.stale:
                manager.canvas.draw_idle()


atexit.register(Gcf.destroy_all)
