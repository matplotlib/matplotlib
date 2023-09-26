from ._tritools import *  # noqa: F401, F403

raise ValueError(f"Importing {__name__} was deprecated in Matplotlib 3.7 and will
                be removed two minor releases later. All functionality is 
                available via the top-level module matplotlib.tri")
                