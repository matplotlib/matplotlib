"""
=================
Subfigure mosaic
=================

This example is inspired by the `subplot_mosaic()
<https://matplotlib.org/devdocs/users/explain/axes/mosaic.html>`__
and `subfigures()
<https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subfigures.html>`__
examples. It especially aims to mimic the former. Most of the API
that is going to be described below is analogous to the `.Figure.subplot_mosaic` API.

`.Figure.subfigure_mosaic` provides a simple way of constructing
complex layouts, such as SubFigures that span multiple columns / rows
of the layout or leave some areas of the Figure blank. The layouts are
constructed either through ASCII art or nested lists.

This interface naturally supports naming your SubFigures. `.Figure.subfigure_mosaic`
returns a dictionary keyed on the labels used to lay out the Figure.
"""
import matplotlib.pyplot as plt
import numpy as np


# Let's define a function to help visualize
def identify_subfigs(subfig_dict, fontsize=36):
    """
    Helper to identify the SubFigures in the examples below.

    Draws the label in a large font in the center of the SubFigure.

    Parameters
    ----------
    subfig_dict : dict[str, SubFigure]
        Mapping between the title / label and the SubFigures.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, subfig in subfig_dict.items():
        subfig.text(0.5, 0.5, k, **kw)


# %%
# If we want a 2x2 grid we can use `.Figure.subfigures` which returns a 2D array
# of `.figure.SubFigure` which we can index.

fig = plt.figure()
subfigs = fig.subfigures(2, 2)

subfigs[0, 0].set_edgecolor('black')
subfigs[0, 0].set_linewidth(2.1)
subfigs[1, 1].set_edgecolor('yellow')
subfigs[1, 1].set_linewidth(2.1)
subfigs[0, 1].set_facecolor('green')

identify_subfigs(
    {(j, k): a for j, r in enumerate(subfigs) for k, a in enumerate(r)},
)

# %%
# Using `.Figure.subfigure_mosaic` we can produce the same mosaic but give the
# SubFigures names

fig = plt.figure()
subfigs = fig.subfigure_mosaic(
    [
        ["First", "Second"],
        ["Third", "Fourth"],
    ],
)
subfigs["First"].set_edgecolor('black')
subfigs["First"].set_linewidth(2.1)
subfigs["Second"].set_facecolor('green')
subfigs["Fourth"].set_edgecolor('yellow')
subfigs["Fourth"].set_linewidth(2.1)

identify_subfigs(subfigs)

# %%
# A key difference between `.Figure.subfigures` and
# `.Figure.subfigure_mosaic` is the return value. While the former
# returns an array for index access, the latter returns a dictionary
# mapping the labels to the `.figure.SubFigure` instances created

print(subfigs)

# %%
# String short-hand
# =================
#
# By restricting our labels to single characters we can
# "draw" the SubFigures we want as "ASCII art".  The following


mosaic = """
    AB
    CD
    """

# %%
# will give us 4 SubFigures laid out in a 2x2 grid and generate the same
# subfigure mosaic as above (but now labeled with ``{"A", "B", "C",
# "D"}`` rather than ``{"First", "Second", "Third", "Fourth"}``).
# Bear in mind that subfigures do not come 'visible' the way subplots do.
# In case you want them to be clearly visible - you will need to set certain
# keyword arguments (such as edge/face color). This is discussed at length in the
# :ref:`controlling-creation` part of this example.

fig = plt.figure()
subfigs = fig.subfigure_mosaic(mosaic)
identify_subfigs(subfigs)

# %%
# Alternatively, you can use the more compact string notation:
mosaic = "AB;CD"

# %%
# will give you the same composition, where the ``";"`` is used
# as the row separator instead of newline.

fig = plt.figure()
subfigs = fig.subfigure_mosaic(mosaic)
identify_subfigs(subfigs)

# %%
# SubFigures spanning multiple rows/columns
# =========================================
#
# Something we can do with `.Figure.subfigure_mosaic`, that we cannot
# do with `.Figure.subfigures`, is to specify that a SubFigure should span
# several rows or columns.


# %%
# If we want to re-arrange our four SubFigures to have ``"C"`` be a horizontal
# span on the bottom and ``"D"`` be a vertical span on the right we would do:

subfigs = plt.figure().subfigure_mosaic(
    """
    ABD
    CCD
    """
)

# setting edges for clarity
for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)

identify_subfigs(subfigs)

# %%
# If we do not want to fill in all the spaces in the Figure with SubFigures,
# we can specify some spaces in the grid to be blank, like so:

subfigs = plt.figure().subfigure_mosaic(
    """
    A.C
    BBB
    .D.
    """
)

# setting edges for clarity
for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)

identify_subfigs(subfigs)

# %%
# If we prefer to use another character (rather than a period ``"."``)
# to mark the empty space, we can use *empty_sentinel* to specify the
# character to use.

subfigs = plt.figure().subfigure_mosaic(
    """
    aX
    Xb
    """,
    empty_sentinel="X",
)

# setting edges for clarity
for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)
identify_subfigs(subfigs)

# %%
#
# Internally there is no meaning attached to the letters we use, any
# Unicode code point is valid!

subfigs = plt.figure().subfigure_mosaic(
    """αя
       ℝ☢"""
)

# setting edges for clarity
for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)
identify_subfigs(subfigs)

# %%
# It is not recommended to use white space as either a label or an
# empty sentinel with the string shorthand because it may be stripped
# while processing the input.
#
# Controlling mosaic creation
# ===========================
#
# This feature is built on top of `.gridspec` and you can pass the
# keyword arguments through to the underlying `.gridspec.GridSpec`
# (the same as `.Figure.subfigures`).
#
# In this case we want to use the input to specify the arrangement,
# but set the relative widths of the rows / columns.  For convenience,
# `.gridspec.GridSpec`'s *height_ratios* and *width_ratios* are exposed in the
# `.Figure.subfigure_mosaic` calling sequence.

subfigs = plt.figure().subfigure_mosaic(
    """
    .a.
    bAc
    .d.
    """,
    # set the height ratios between the rows
    height_ratios=[1, 3.5, 1],
    # set the width ratios between the columns
    width_ratios=[1, 3.5, 1],
)

# setting edges for clarity
for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)
identify_subfigs(subfigs)

# %%
# You can also use the `.Figure.subfigures` functionality to
# position the overall mosaic to put multiple versions of the same
# mosaic in a figure.

mosaic = """AA
            BC"""
fig = plt.figure()

left, right = fig.subfigures(nrows=1, ncols=2)

subfigs = left.subfigure_mosaic(mosaic)
for subfig in subfigs.values():
    subfig.set_edgecolor('black')
    subfig.set_linewidth(2.1)
identify_subfigs(subfigs)

subfigs = right.subfigure_mosaic(mosaic)
for subfig in subfigs.values():
    subfig.set_edgecolor('black')
    subfig.set_linewidth(2.1)
identify_subfigs(subfigs)

# %%
# .. _controlling-creation:
# Controlling subfigure creation
# ==============================
#
# We can also pass through arguments used to create the subfigures
# which will apply to all of the SubFigures created. So instead of iterating like so:

for sf in subfigs.values():
    sf.set_edgecolor('black')
    sf.set_linewidth(2.1)

# %%
# we would write:

subfigs = plt.figure().subfigure_mosaic(
    "A.B;A.C", subfigure_kw={"edgecolor": "black", "linewidth": 2.1}
)
identify_subfigs(subfigs)

# %%
# Per-SubFigure keyword arguments
# ----------------------------------
#
# If you need to control the parameters passed to each subfigure individually use
# *per_subfigure_kw* to pass a mapping between the SubFigure identifiers (or
# tuples of SubFigure identifiers) to dictionaries of keywords to be passed.
#

fig, subfigs = plt.subfigure_mosaic(
    "AB;CD",
    per_subfigure_kw={
        "A": {"facecolor": "green"},
        ("C", "D"): {"edgecolor": "black", "linewidth": 1.1, }
    },
)
identify_subfigs(subfigs)

# %%
# If the layout is specified with the string short-hand, then we know the
# SubFigure labels will be one character and can unambiguously interpret longer
# strings in *per_subfigure_kw* to specify a set of SubFigures to apply the
# keywords to:

fig, subfigs = plt.subfigure_mosaic(
    "AB;CD",
    per_subfigure_kw={
        "AD": {"facecolor": ".3"},
        "BC": {"edgecolor": "black", "linewidth": 2.1, }
    },
)
identify_subfigs(subfigs)

# %%
# If *subfigure_kw* and *per_subfigure_kw* are used together, then they are
# merged with *per_subfigure_kw* taking priority:

subfigs = plt.figure().subfigure_mosaic(
    "AB;CD",
    subfigure_kw={"facecolor": "xkcd:tangerine", "linewidth": 2},
    per_subfigure_kw={
        "B": {"facecolor": "xkcd:water blue"},
        "D": {"edgecolor": "yellow", "linewidth": 2.2, "facecolor": "g"},
    }
)
identify_subfigs(subfigs)

# %%
# Nested list input
# =================
#
# Everything we can do with the string shorthand we can also do when
# passing in a list (internally we convert the string shorthand to a nested
# list), for example using spans and blanks:

subfigs = plt.figure().subfigure_mosaic(
    [
        ["main", "zoom"],
        ["main", "BLANK"],
    ],
    empty_sentinel="BLANK",
    width_ratios=[2, 1],
    subfigure_kw={"facecolor": "xkcd:sea green", "linewidth": 2},
)
identify_subfigs(subfigs)

# %%
# In addition, using the list input we can specify nested mosaics.  Any element
# of the inner list can be another set of nested lists:

inner = [
    ["inner A"],
    ["inner B"],
]
inner_three = [
    ["inner Q"],
    ["inner Z"],
]
inner_two = [
    ["inner C"],
    [inner_three],
]

layout = [["A", [[inner_two, "C"],
                 ["D", "E"]]
           ],
          ["F", "G"],
          [".", [["H", [["I"],
                        ["."]
                        ]
                  ]
                 ]
           ]
          ]
fig, subfigs = plt.subfigure_mosaic(layout, subfigure_kw={'edgecolor': 'black',
                                                          'linewidth': 1.5},
                                    per_subfigure_kw={"E": {'edgecolor': 'xkcd:red'},
                                                      "G": {'facecolor': 'yellow'},
                                                      "H": {'edgecolor': 'blue',
                                                            'facecolor': 'xkcd:azure'}}
                                    )

identify_subfigs(subfigs, fontsize=12)

# %%
# We can also pass in a 2D NumPy array to do things like:
mosaic = np.zeros((4, 4), dtype=int)
for j in range(4):
    mosaic[j, j] = j + 1
subfigs = plt.figure().subfigure_mosaic(
    mosaic,
    subfigure_kw={'edgecolor': 'black', 'linewidth': 1.5},
    empty_sentinel=0,
)
identify_subfigs(subfigs, fontsize=12)
