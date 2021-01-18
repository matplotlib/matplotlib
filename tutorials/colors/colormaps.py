"""
********************************
Choosing Colormaps in Matplotlib
********************************

Matplotlib has a number of built-in colormaps accessible via
`.matplotlib.cm.get_cmap`.  There are also external libraries like
[palettable]_ and [colorcet]_ that have many extra colormaps.
Here we briefly discuss how to choose between the many options.  For
help on creating your own colormaps, see
:doc:`/tutorials/colors/colormap-manipulation`.

Overview
========

The idea behind choosing a good colormap is to find a good representation in 3D
colorspace for your data set. The best colormap for any given data set depends
on many things including:

- Whether representing form or metric data ([Ware]_)

- Your knowledge of the data set (*e.g.*, is there a critical value
  from which the other values deviate?)

- If there is an intuitive color scheme for the parameter you are plotting

- If there is a standard in the field the audience may be expecting

For many applications, a perceptually uniform colormap is the best choice;
i.e. a colormap in which equal steps in data are perceived as equal
steps in the color space. Researchers have found that the human brain
perceives changes in the lightness parameter as changes in the data
much better than, for example, changes in hue. Therefore, colormaps
which have monotonically increasing lightness through the colormap
will be better interpreted by the viewer. A wonderful example of
perceptually uniform colormaps is [colorcet]_.

Color can be represented in 3D space in various ways. One way to represent color
is using CIELAB. In CIELAB, color space is represented by lightness,
:math:`L^*`; red-green, :math:`a^*`; and yellow-blue, :math:`b^*`. The lightness
parameter :math:`L^*` can then be used to learn more about how the matplotlib
colormaps will be perceived by viewers.

An excellent starting resource for learning about human perception of colormaps
is from [IBM]_.


Classes of colormaps
====================

Colormaps are often split into several categories based on their function (see,
*e.g.*, [Moreland]_):

1. Sequential: change in lightness and often saturation of color
   incrementally, often using a single hue; should be used for
   representing information that has ordering.

2. Diverging: change in lightness and possibly saturation of two
   different colors that meet in the middle at an unsaturated color;
   should be used when the information being plotted has a critical
   middle value, such as topography or when the data deviates around
   zero.

3. Cyclic: change in lightness of two different colors that meet in
   the middle and beginning/end at an unsaturated color; should be
   used for values that wrap around at the endpoints, such as phase
   angle, wind direction, or time of day.

4. Qualitative: often are miscellaneous colors; should be used to
   represent information which does not have ordering or
   relationships.
"""

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

from matplotlib.colors import ListedColormap
turbo_colormap_data = [[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024], [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314], [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096], [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373], [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142], [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406], [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162], [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412], [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156], [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393], [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123], [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347], [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064], [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275], [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930], [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773], [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835], [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202], [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959], [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190], [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981], [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416], [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580], [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559], [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437], [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299], [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230], [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316], [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599], [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627], [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308], [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713], [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914], [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981], [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987], [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002], [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098], [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345], [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816], [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581], [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712], [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280], [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [
    0.64362, 0.98999, 0.23356], [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960], [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032], [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504], [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311], [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386], [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663], [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074], [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555], [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038], [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456], [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744], [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836], [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663], [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161], [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358], [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348], [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165], [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842], [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412], [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910], [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367], [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817], [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294], [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831], [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461], [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218], [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134], [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243], [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521], [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875], [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305], [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809], [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387], [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041], [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769], [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571], [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449], [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401], [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427], [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529], [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705], [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955], [0.47960, 0.01583, 0.01055]]
cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))

cmaps = OrderedDict()

###############################################################################
# Sequential
# ----------
#
# For the Sequential plots, the lightness value increases monotonically through
# the colormaps. This is good. Some of the :math:`L^*` values in the colormaps
# span from 0 to 100 (binary and the other grayscale), and others start around
# :math:`L^*=20`. Those that have a smaller range of :math:`L^*` will accordingly
# have a smaller perceptual range. Note also that the :math:`L^*` function varies
# amongst the colormaps: some are approximately linear in :math:`L^*` and others
# are more curved.

cmaps['Perceptually Uniform Sequential'] = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

###############################################################################
# Sequential2
# -----------
#
# Many of the :math:`L^*` values from the Sequential2 plots are monotonically
# increasing, but some (autumn, cool, spring, and winter) plateau or even go both
# up and down in :math:`L^*` space. Others (afmhot, copper, gist_heat, and hot)
# have kinks in the :math:`L^*` functions. Data that is being represented in a
# region of the colormap that is at a plateau or kink will lead to a perception of
# banding of the data in those values in the colormap (see [mycarta-banding]_ for
# an excellent example of this).

cmaps['Sequential (2)'] = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper']

###############################################################################
# Diverging
# ---------
#
# For the Diverging maps, we want to have monotonically increasing :math:`L^*`
# values up to a maximum, which should be close to :math:`L^*=100`, followed by
# monotonically decreasing :math:`L^*` values. We are looking for approximately
# equal minimum :math:`L^*` values at opposite ends of the colormap. By these
# measures, BrBG and RdBu are good options. coolwarm is a good option, but it
# doesn't span a wide range of :math:`L^*` values (see grayscale section below).

cmaps['Diverging'] = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

###############################################################################
# Cyclic
# ------
#
# For Cyclic maps, we want to start and end on the same color, and meet a
# symmetric center point in the middle. :math:`L^*` should change monotonically
# from start to middle, and inversely from middle to end. It should be symmetric
# on the increasing and decreasing side, and only differ in hue. At the ends and
# middle, :math:`L^*` will reverse direction, which should be smoothed in
# :math:`L^*` space to reduce artifacts. See [kovesi-colormaps]_ for more
# information on the design of cyclic maps.
#
# The often-used HSV colormap is included in this set of colormaps, although it
# is not symmetric to a center point. Additionally, the :math:`L^*` values vary
# widely throughout the colormap, making it a poor choice for representing data
# for viewers to see perceptually. See an extension on this idea at
# [mycarta-jet]_.

cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

###############################################################################
# Qualitative
# -----------
#
# Qualitative colormaps are not aimed at being perceptual maps, but looking at the
# lightness parameter can verify that for us. The :math:`L^*` values move all over
# the place throughout the colormap, and are clearly not monotonically increasing.
# These would not be good options for use as perceptual colormaps.

cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']

###############################################################################
# Miscellaneous
# -------------
#
# Some of the miscellaneous colormaps have particular uses for which
# they have been created. For example, gist_earth, ocean, and terrain
# all seem to be created for plotting topography (green/brown) and water
# depths (blue) together. We would expect to see a divergence in these
# colormaps, then, but multiple kinks may not be ideal, such as in
# gist_earth and terrain. CMRmap was created to convert well to
# grayscale, though it does appear to have some small kinks in
# :math:`L^*`.  cubehelix was created to vary smoothly in both lightness
# and hue, but appears to have a small hump in the green hue area. turbo
# was created to display depth and disparity data.
#
# The often-used jet colormap is included in this set of colormaps. We can see
# that the :math:`L^*` values vary widely throughout the colormap, making it a
# poor choice for representing data for viewers to see perceptually. See an
# extension on this idea at [mycarta-jet]_ and [turbo]_.

cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']

###############################################################################
# .. _color-colormaps_reference:
#
# First, we'll show the range of each colormap. Note that some seem
# to change more "quickly" than others.

nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axs = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.1, bottom=0.1, left=0.2, right=0.99)
    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps.items():
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()

###############################################################################
# Lightness of Matplotlib colormaps
# =================================
#
# Here we examine the lightness values of the matplotlib colormaps.
# Note that some documentation on the colormaps is available
# ([list-colormaps]_).

mpl.rcParams.update({'font.size': 12})

# Number of colormap per subplot for particular cmap categories
_DSUBS = {'Perceptually Uniform Sequential': 5, 'Sequential': 6,
          'Sequential (2)': 6, 'Diverging': 6, 'Cyclic': 3,
          'Qualitative': 4, 'Miscellaneous': 6}

# Spacing between the colormaps of a subplot
_DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
       'Sequential (2)': 1.4, 'Diverging': 1.4, 'Cyclic': 1.4,
       'Qualitative': 1.4, 'Miscellaneous': 1.4}

# Indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
for cmap_category, cmap_list in cmaps.items():

    # Do subplots so that colormaps have enough space.
    # Default is 6 colormaps per subplot.
    dsub = _DSUBS.get(cmap_category, 6)
    nsubplots = int(np.ceil(len(cmap_list) / dsub))

    # squeeze=False to handle similarly the case of a single subplot
    fig, axs = plt.subplots(nrows=nsubplots, squeeze=False,
                            figsize=(7, 2.6 * nsubplots))

    for i, ax in enumerate(axs.flat):

        locs = []  # locations for text labels

        for j, cmap in enumerate(cmap_list[i * dsub:(i + 1) * dsub]):

            # Get RGB values for colormap and convert the colormap in
            # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
            rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
            lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

            # Plot colormap L values.  Do separately for each category
            # so each plot can be pretty.  To make scatter markers change
            # color along plot:
            # http://stackoverflow.com/questions/8202605/

            if cmap_category == 'Sequential':
                # These colormaps all start at high lightness but we want them
                # reversed to look nice in the plot, so reverse the order.
                y_ = lab[0, ::-1, 0]
                c_ = x[::-1]
            else:
                y_ = lab[0, :, 0]
                c_ = x

            dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing
            ax.scatter(x + j * dc, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)

            # Store locations for colormap labels
            if cmap_category in ('Perceptually Uniform Sequential',
                                 'Sequential'):
                locs.append(x[-1] + j * dc)
            elif cmap_category in ('Diverging', 'Qualitative', 'Cyclic',
                                   'Miscellaneous', 'Sequential (2)'):
                locs.append(x[int(x.size / 2.)] + j * dc)

        # Set up the axis limits:
        #   * the 1st subplot is used as a reference for the x-axis limits
        #   * lightness values goes from 0 to 100 (y-axis limits)
        ax.set_xlim(axs[0, 0].get_xlim())
        ax.set_ylim(0.0, 100.0)

        # Set up labels for colormaps
        ax.xaxis.set_ticks_position('top')
        ticker = mpl.ticker.FixedLocator(locs)
        ax.xaxis.set_major_locator(ticker)
        formatter = mpl.ticker.FixedFormatter(
            cmap_list[i * dsub:(i + 1) * dsub])
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel('Lightness $L^*$', fontsize=12)

    ax.set_xlabel(cmap_category + ' colormaps', fontsize=14)

    fig.tight_layout(h_pad=0.0, pad=1.5)
    plt.show()


###############################################################################
# Grayscale conversion
# ====================
#
# It is important to pay attention to conversion to grayscale for color
# plots, since they may be printed on black and white printers.  If not
# carefully considered, your readers may end up with indecipherable
# plots because the grayscale changes unpredictably through the
# colormap.
#
# Conversion to grayscale is done in many different ways [bw]_. Some of the
# better ones use a linear combination of the rgb values of a pixel, but
# weighted according to how we perceive color intensity. A nonlinear method of
# conversion to grayscale is to use the :math:`L^*` values of the pixels. In
# general, similar principles apply for this question as they do for presenting
# one's information perceptually; that is, if a colormap is chosen that is
# monotonically increasing in :math:`L^*` values, it will print in a reasonable
# manner to grayscale.
#
# With this in mind, we see that the Sequential colormaps have reasonable
# representations in grayscale. Some of the Sequential2 colormaps have decent
# enough grayscale representations, though some (autumn, spring, summer,
# winter) have very little grayscale change. If a colormap like this was used
# in a plot and then the plot was printed to grayscale, a lot of the
# information may map to the same gray values. The Diverging colormaps mostly
# vary from darker gray on the outer edges to white in the middle. Some
# (PuOr and seismic) have noticeably darker gray on one side than the other
# and therefore are not very symmetric. coolwarm has little range of gray scale
# and would print to a more uniform plot, losing a lot of detail. Note that
# overlaid, labeled contours could help differentiate between one side of the
# colormap vs. the other since color cannot be used once a plot is printed to
# grayscale. Many of the Qualitative and Miscellaneous colormaps, such as
# Accent, hsv, jet and turbo, change from darker to lighter and back to darker
# grey throughout the colormap. This would make it impossible for a viewer to
# interpret the information in a plot once it is printed in grayscale.

mpl.rcParams.update({'font.size': 14})

# Indices to step through colormap.
x = np.linspace(0.0, 1.0, 100)

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    fig, axs = plt.subplots(nrows=len(cmap_list), ncols=2)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
                        wspace=0.05)
    fig.suptitle(cmap_category + ' colormaps', fontsize=14, y=1.0, x=0.6)

    for ax, name in zip(axs, cmap_list):

        # Get RGB values for colormap.
        rgb = cm.get_cmap(plt.get_cmap(name))(x)[np.newaxis, :, :3]

        # Get colormap in CAM02-UCS colorspace. We want the lightness.
        lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        L = lab[0, :, 0]
        L = np.float32(np.vstack((L, L, L)))

        ax[0].imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
        pos = list(ax[0].get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs.flat:
        ax.set_axis_off()

    plt.show()


for cmap_category, cmap_list in cmaps.items():

    plot_color_gradients(cmap_category, cmap_list)

###############################################################################
# Color vision deficiencies
# =========================
#
# There is a lot of information available about color blindness (*e.g.*,
# [colorblindness]_). Additionally, there are tools available to convert images
# to how they look for different types of color vision deficiencies.
#
# The most common form of color vision deficiency involves differentiating
# between red and green. Thus, avoiding colormaps with both red and green will
# avoid many problems in general.
#
#
# References
# ==========
#
# .. [colorcet] https://colorcet.pyviz.org
# .. [Ware] http://ccom.unh.edu/sites/default/files/publications/Ware_1988_CGA_Color_sequences_univariate_maps.pdf
# .. [Moreland] http://www.kennethmoreland.com/color-maps/ColorMapsExpanded.pdf
# .. [list-colormaps] https://gist.github.com/endolith/2719900#id7
# .. [mycarta-banding] https://mycarta.wordpress.com/2012/10/14/the-rainbow-is-deadlong-live-the-rainbow-part-4-cie-lab-heated-body/
# .. [mycarta-jet] https://mycarta.wordpress.com/2012/10/06/the-rainbow-is-deadlong-live-the-rainbow-part-3/
# .. [kovesi-colormaps] https://arxiv.org/abs/1509.03700
# .. [bw] http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
# .. [colorblindness] http://www.color-blindness.com/
# .. [IBM] https://doi.org/10.1109/VISUAL.1995.480803
# .. [palettable] https://jiffyclub.github.io/palettable/
# .. [turbo] https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
