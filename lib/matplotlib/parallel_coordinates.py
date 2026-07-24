"""
Parallel coordinates plotting module.

A parallel coordinate plot visualizes multi-dimensional data by drawing
vertical axes for each data dimension and connecting data points as
polylines across these axes. This is useful for exploring patterns and
relationships in high-dimensional datasets.
"""

import numpy as np

from matplotlib import _api
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

__all__ = ['parallel_coordinates']


def _get_data_values(data, cols):
    """Extract numeric columns from *data* and return array + column names."""
    if hasattr(data, 'columns'):
        if cols is None:
            cols = []
            for c in data.columns:
                try:
                    if np.issubdtype(data[c].dtype, np.number):
                        cols.append(c)
                except (TypeError, ValueError):
                    try:
                        data[c].astype(float)
                        cols.append(c)
                    except (ValueError, TypeError):
                        pass
        elif hasattr(cols[0], '__index__'):
            data_cols = list(data.columns)
            cols = [data_cols[i] for i in cols]
        df = data[cols]
        values = df.to_numpy(dtype=float, na_value=np.nan)
        col_names = list(df.columns)
    else:
        data_arr = np.asarray(data)
        if data_arr.ndim != 2:
            raise ValueError("data must be 2-dimensional")
        if cols is None:
            cols = list(range(data_arr.shape[1]))
            subset = data_arr[:, cols]
            col_names = [str(c) for c in cols]
        elif hasattr(cols[0], '__index__'):
            subset = data_arr[:, cols]
            col_names = [str(c) for c in cols]
        else:
            subset = data_arr
            col_names = list(cols)
            if subset.shape[1] != len(cols):
                raise ValueError(
                    f"Number of column names ({len(cols)}) does not match "
                    f"number of data columns ({subset.shape[1]})")
        try:
            values = subset.astype(float)
        except (ValueError, TypeError):
            raise ValueError(
                "Selected columns must be numeric; "
                "use cols= to exclude non-numeric columns")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    return values, col_names


def parallel_coordinates(axes, data, class_column=None, cols=None,
                         color=None, cmap=None, alpha=0.5,
                         linewidth=1, linestyle='-'):
    """
    Draw a parallel coordinates plot.

    A parallel coordinate plot allows visualization of multi-dimensional
    data by drawing vertical axes for each feature and connecting
    individual data points as polylines.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The axes to draw on.

    data : DataFrame or (N, M) array-like
        The data to visualize. If a DataFrame is provided, column names
        are used as axis labels.

    class_column : str or int, optional
        Column name or index used to color lines by class. Each unique
        value in this column gets a different color.

    cols : list of str or int, optional
        Columns to include in the plot. If None, all numeric columns are
        used.

    color : :mpltype:`color` or list of :mpltype:`color`, optional
        Colors for each class (if *class_column* is given) or a single
        color for all lines. If a list, must match the number of unique
        classes.

    cmap : str or `~matplotlib.colors.Colormap`, optional
        Colormap used to color lines by class. Ignored if *color* is
        provided.

    alpha : float, default 0.5
        Transparency of the lines.

    linewidth : float, default 1
        Width of the lines.

    linestyle : str, default '-'
        Style of the lines.

    Returns
    -------
    list of `.LineCollection`
        List of `.LineCollection` objects, one per class (a single-element
        list if no *class_column* is given).
    """
    values, col_names = _get_data_values(data, cols)

    n_dims = values.shape[1]
    if n_dims < 2:
        raise ValueError(
            "parallel_coordinates requires at least 2 dimensions, "
            f"got {n_dims}")

    if class_column is not None:
        if hasattr(data, 'columns'):
            class_values = data[class_column]
            class_values = np.asarray(class_values)
        else:
            class_values = np.asarray(data[:, class_column])
        classes, class_idx = np.unique(class_values, return_inverse=True)
        n_classes = len(classes)
    else:
        classes = [None]
        class_idx = np.zeros(values.shape[0], dtype=int)
        n_classes = 1

    vmin = np.min(values, axis=0)
    vmax = np.max(values, axis=0)
    ranges = vmax - vmin
    ranges[ranges == 0] = 1
    normalized = (values - vmin) / ranges

    if color is not None:
        from matplotlib.colors import is_color_like
        if isinstance(color, str) and is_color_like(color):
            color_list = [color] * n_classes
        elif isinstance(color, (tuple, list)) and len(color) > 0 \
                and all(is_color_like(c) for c in color):
            color_list = list(color)
            if len(color_list) == 1:
                color_list = color_list * n_classes
        else:
            color_list = list(color)
        if len(color_list) != n_classes:
            _api.warn_external(
                f"color list length ({len(color_list)}) does not match "
                f"number of classes ({n_classes}); will cycle")
            color_list = (color_list * (n_classes // len(color_list) + 1)
                          )[:n_classes]
    elif cmap is not None:
        from matplotlib import colormaps
        cmap_obj = colormaps[cmap] if isinstance(cmap, str) else cmap
        if n_classes == 1:
            color_list = [cmap_obj(0.5)]
        else:
            color_list = [cmap_obj(i / (n_classes - 1))
                          for i in range(n_classes)]
    else:
        color_list = [axes._get_lines.get_next_color()
                      for _ in range(n_classes)]

    x = np.arange(n_dims, dtype=float)
    xmin, xmax = x[0] - 0.5, x[-1] + 0.5

    collections = []
    legend_handles = []

    for cls_idx in range(n_classes):
        mask = class_idx == cls_idx
        if not np.any(mask):
            continue
        cls_data = normalized[mask]
        segments = np.empty((cls_data.shape[0], n_dims, 2))
        segments[:, :, 0] = x
        segments[:, :, 1] = cls_data

        lc = LineCollection(
            segments, colors=color_list[cls_idx],
            alpha=alpha, linewidth=linewidth,
            linestyles=linestyle, zorder=2)
        axes.add_collection(lc)
        collections.append(lc)

        if class_column is not None:
            legend_handles.append(
                Line2D([0], [0], color=color_list[cls_idx],
                       linewidth=linewidth, linestyle=linestyle,
                       label=str(classes[cls_idx])))

    for i, name in enumerate(col_names):
        axes.axvline(x[i], color='grey', linestyle='--',
                     linewidth=0.5, zorder=1)

    axes.set_xlim(xmin, xmax)
    axes.set_ylim(-0.05, 1.05)
    axes.set_xticks(x)
    axes.set_xticklabels(col_names, rotation=45, ha='right')
    axes.set_ylabel('Normalized range')
    axes.set_frame_on(False)
    axes.xaxis.set_ticks_position('none')

    if class_column is not None and legend_handles:
        axes.legend(handles=legend_handles, loc='best')

    axes.autoscale_view()

    return collections
