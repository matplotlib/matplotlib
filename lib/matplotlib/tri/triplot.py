from ._triplot import *  # noqa: F401, F403

raise ValueError(f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
                f"be removed two minor releases later. All functionality is "
                f"available via the top-level module matplotlib.tri")