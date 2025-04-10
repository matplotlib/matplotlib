"""Locally patch Figure to accept a label kwarg."""


from matplotlib.figure import Figure as _Figure


class Figure(_Figure):
    """Thin sub-class of Figure to accept a label on init."""

    def __init__(self, *args, label=None, **kwargs):
        # docstring inherited
        super().__init__(*args, **kwargs)
        if label is not None:
            self.set_label(label)
