"""
Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
answer:
https://stackoverflow.com/q/2225995/

(https://stackoverflow.com/users/66549/doug)
"""

import itertools

import numpy as np

from matplotlib import _api

__all__ = ['stackplot']


def stackplot(axes, x, *args,
              labels=(), colors=None, hatch=None, baseline='zero',
              **kwargs):
    """
    Draw a stacked area plot or a streamgraph.

    Parameters
    ----------
    x : array-like

    y : multiple array-like, 2D array-like or pandas.DataFrame

        - multiple array-like: the data is unstacked
        
        .. code-block:: none

                  #     year_1,    year_2,    year_3
                  y1 = [value_1_A, value_2_A, value_3_A] # category_A
                  y2 = [value_1_B, value_2_B, value_3_B] # category_B
                  y3 = [value_1_C, value_2_C, value_3_C] # category_C
        
                  x = [*range(3)]

        Example call:

        .. code-block:: python

            stackplot(x, y1, y2, y3)

        - 2D array-like: Each row represents a category, each column represents an x-axis dimension associated with the categories. A list of 1D array-like can also be passed; each list item must have the same length

        Example call:

        .. code-block:: python
        
            y = [y1, y2, y3]
            x = [*range(3)]

            stackplot(x, y)    

        - a `pandas.DataFrame`: The index is used for the categories, each column represents an x-axis dimension associated with the categories.

        Example call:

        .. code-block:: python

            y = pd.DataFrame(
                        np.random.random((3, 3)),
                        index=["category_A", "category_B", "category_C"],
                        columns=[*range(3)]
                    )
            
            x = df.columns

            stackplot(x, y)

    baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        Method used to calculate the baseline:

        - ``'zero'``: Constant zero baseline, i.e. a simple stacked plot.
        - ``'sym'``:  Symmetric around zero and is sometimes called
          'ThemeRiver'.
        - ``'wiggle'``: Minimizes the sum of the squared slopes.
        - ``'weighted_wiggle'``: Does the same but weights to account for
          size of each layer. It is also called 'Streamgraph'-layout. More
          details can be found at http://leebyron.com/streamgraph/.

    labels : list of str, optional
        A sequence of labels to assign to each data series. If unspecified,
        then no labels will be applied to artists.

    colors : list of :mpltype:`color`, optional
        A sequence of colors to be cycled through and used to color the stacked
        areas. The sequence need not be exactly the same length as the number
        of provided *y*, in which case the colors will repeat from the
        beginning.

        If not specified, the colors from the Axes property cycle will be used.

    hatch : list of str, default: None
        A sequence of hatching styles.  See
        :doc:`/gallery/shapes_and_collections/hatch_style_reference`.
        The sequence will be cycled through for filling the
        stacked areas from bottom to top.
        It need not be exactly the same length as the number
        of provided *y*, in which case the styles will repeat from the
        beginning.

        .. versionadded:: 3.9
           Support for list input

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between`.

    Returns
    -------
    list of `.PolyCollection`
        A list of `.PolyCollection` instances, one for each element in the
        stacked area plot.
    """

    y = np.vstack(args)

    labels = iter(labels)
    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = (axes._get_lines.get_next_color() for _ in y)

    if hatch is None or isinstance(hatch, str):
        hatch = itertools.cycle([hatch])
    else:
        hatch = itertools.cycle(hatch)

    # Assume data passed has not been 'stacked', so stack it here.
    # We'll need a float buffer for the upcoming calculations.
    stack = np.cumsum(y, axis=0, dtype=np.promote_types(y.dtype, np.float32))

    _api.check_in_list(['zero', 'sym', 'wiggle', 'weighted_wiggle'],
                       baseline=baseline)
    if baseline == 'zero':
        first_line = 0.

    elif baseline == 'sym':
        first_line = -np.sum(y, 0) * 0.5
        stack += first_line[None, :]

    elif baseline == 'wiggle':
        m = y.shape[0]
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        stack += first_line

    elif baseline == 'weighted_wiggle':
        total = np.sum(y, 0)
        # multiply by 1/total (or zero) to avoid infinities in the division:
        inv_total = np.zeros_like(total)
        mask = total > 0
        inv_total[mask] = 1.0 / total[mask]
        increase = np.hstack((y[:, 0:1], np.diff(y)))
        below_size = total - stack
        below_size += 0.5 * y
        move_up = below_size * inv_total
        move_up[:, 0] = 0.5
        center = (move_up - 0.5) * increase
        center = np.cumsum(center.sum(0))
        first_line = center - 0.5 * total
        stack += first_line

    # Color between x = 0 and the first array.
    coll = axes.fill_between(x, first_line, stack[0, :],
                             facecolor=next(colors),
                             hatch=next(hatch),
                             label=next(labels, None),
                             **kwargs)
    coll.sticky_edges.y[:] = [0]
    r = [coll]

    # Color between array i-1 and array i
    for i in range(len(y) - 1):
        r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
                                   facecolor=next(colors),
                                   hatch=next(hatch),
                                   label=next(labels, None),
                                   **kwargs))
    return r
