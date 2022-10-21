import numpy as np
import matplotlib.colors
import matplotlib.axes
from matplotlib.cm import get_cmap
import math


def bivariate_legend(ax, data1, data2, cmap='red_blue', d1norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                    d2norm=matplotlib.colors.Normalize(vmin=0, vmax=1), d1num_bins=1, d2num_bins=1, mode='both', d1category_labels=None,
                    d2category_labels=None, darkenWhite=False, title=None, title_fontproperties=None, label_fontproperties=None,
                    legend_loc=None, legend_padding=None, d1label=None, d2label=None, d1minimum=None, d1maximum=None,
                    d2minimum=None, d2maximum=None, d1ticks=None, d2ticks=None, d1extend=None, d1extendfrac=None, d2extend=None,
                    d2extendfrac=None, d1tick_rotation=None, d2tick_rotation=None, d1bins=None, d2bins=None, subplot_adjust=None,
                    **imshow_kwargs):
    """
    Create bivariate colormap and corresponding legend.

    Being restricted to mapping data to a univariate colormap can often limit the information that can be extracted from datasets. 
    The ability of mapping the data based on 2 variables can uncover new correlations, valuable grouping in datasets, and important trends.
    When mapping in a bivariate manner, each variable of a data point is mapped to their respective univariate color scale. 
    Then, these colors are blended together for a unique bivariate colormapping of the data point, unveiling key information.

    The function also draws a customizable legend image to understand the resulting colormapping. 
    This is constructed using matplotlib's imshow function and with some additional
    legend customization features.

    Parameters
    ----------
    data1 : list or tuple
      First variable (on the x-axis).

    data2 : list or tuple
      Second variable (on the y-axes).

    cmap : {"red_blue",  'purple_cyan', 'yellow_red', 'red_green', 'blue_green', 'yellow_blue', 'pink_blue', 'orange_blue', 'pink_orange'},
    list of str or list of HSV, default: 'red_blue'
      The colormap to be used. Can be one of the 9 bivariateLegend specific colormaps, a list of 2 matplotlib 1D colormaps, 
      or a list of 2 HSV colors where either the saturation or value will be used to create a gradient, depending on the mode setting.

    d1norm : `.colors`, default: `.colors.Normalize(vmin=0, vmax=1)
      Normalization for data1.

    d2norm  : `.colors`, default: `.colors.Normalize(vmin=0, vmax=1)
      Normalization for data2.

    d1num_bins: int, default: 1
      How many bins the data in data1 should be put into. 10 bin limit

    d2num_bins: int, default: 1
      How many bins the data in data2 should be put into. 10 bin limit

    mode : str, default: 'both'
      Whether the color gradients (both x and y) should increase based on value, saturation, or both on the HSV color scale.

    d1category_labels : list of str, optional
      Specified if the user want to name what each bin represents for the data in data1. If there are less category labels than bins
      it will cycle to the front of the list. The tick labels will automatically be rotated by 30°.

    d2category_labels : list of str, optional
      Same purpose but for data2.

    darkenWhite : bool, default: False
      Whether or not the (0,0) coordinate or bin should be made grey if it's supposed to be white. This is useful for being able to easily
      visualize plots.

    title: str, default: "Legend"
      The title for the legend.

    title_fontproperties : `.font_manager.FontProperties`, optional
      An instance of matplotlib's FontProperties class can be passed in to manipulate the font details of the title.

    label_fontproperties : `.font_manager.FontProperties`, optional
      An instance of matplotlib's FontProperties class can be passed in to manipulate the font details of the axis labels.

    legend_loc : [left, bottom, width, height] or Bbox
      The position of the legend location, which will be passed into matplotlib's set_position method.

    legend_padding : int, default: 0.4
      The space between the parent axes and the legend when the position is determined automatically (legend_loc is None). 

    d1label : str, default: "Variable 1"
      The x label for the legend.

    d2label : str, default: "Variable 2"
      The y label for the legend.

    d1minimum : int, optional
      The lower bound for the x axis. Data values in data1 lower than this value will be handeled according to d1extend arguments. 
      This will be disregarded if d1bins is not None.

    d1maximum : int, optional
      The upper bound for the x axis. Data values in data1 higher than this value will be handeled according to d1extend arguments. 
      This will be disregarded if d1bins is not None.

    d2minimium : int, optional
      The lower bound for the y axis. Data values in data2 lower than this value will be handeled according to d2extend arguments. 
      This will be disregarded if d2bins is not None.

    d2maximum : int, optional
      The upper bound for the y axis. Data values in data2 higher than this value will be handeled according to d2extend arguments. 
      This will be disregarded if d2bins is not None.

    d1ticks : dict of `.Axes.set_xticks` keyword arguments, optional
      Paramters for matplotlib's set_xticks method will be unpacks and passed along to this method.

    d2ticks :  dict of `.Axes.set_yticks` keyword arguments, optional
      Paramters for matplotlib's set_yticks method will be unpacks and passed along to this method.

    d1extend : {"both", "min", "max"}, optional
      Whether or not a portion of the color gradient on the x axis will be reserved for out-of-bound values. 
      If None, out-of-bound values in data 1will be mapped to the highest/lowest color. Note that the highest/lowest in-bound value can 
      also be mapped to this color. If "min", only a portion of the color gradient is reserved for values lower than the lower bound 
      (specified by either d1minimum or the first value of d1bins). Same idea goes for if this argument is "max", 
      where a portion is reserved for values higher than the upper bound. If "both", portions will be taken out from both ends.

    d1extendfrac : float, default: 0.05
      Fraction of the end(s) specified to be extended on the x-axis. Only used with d1extend is not None.

    d2extend :  {"both", "min", "max"}, optional
      The same concept as d1extend, but applied to the y-axis/values in data2.

     d2extendfrac : float, default: 0.05
      Fraction of the end(s) specified to be extended on the y-axis. Only used with d2extend is not None.

    d1tickrotation : int, optional
      The number of degrees to rotate the tick labels on the x-axis. 
      Normally, there is no rotation unless d1category_labels is specified, in which case the labels are rotated 30°.

    d2tickrotation : int, optional
      The number of degrees to rotate the tick labels on the y-axis. 
      Normally, there is no rotation unless d2category_labels is specified, in which case the labels are rotated 30°.

    d1bins : list of int or float, optional
      The bins for data1 to be split into. This will be done utilizing np.digitize after appropriate normalization. 
      If this is specified, the first and last values will be used from d1minimum and d1maximum, respectively. 
      Example: [0, 0.25, 0.5, 0.75, 1] Here, values between 0 and 0.25 goes in the first bin, 
      values from 0.25 to 0.5 goes in the second bin, and so on.

    d2bins : list of int or float, optional
      The bins for data2 to be split into. This will be done utilizing np.digitize after appropriate normalization. 
      If this is specified, the first and last values will be used from d2minimum and d2maximum, respectively.

    subplot_adjust : dict of `pyplot.subplots_asjust` keyword arguments, optional
      Key-value pairs will be unpacked and passed along to maptlotlib's `pyplot.subplots_asjust` method.

    **imshow_kwargs 
      imshow properties for the legend.

    Returns
    -------
    colors : list of RGB
      The RGB color mapping of each data point (combination of data1 and data2/each value mix)

    cax : `~.axes.Axes`, or a subclass of `~.axes.Axes`
      The Legend Axes is returned to the user for any further desired customization

    """
    # map the values to a color in the colormap

    # use a pre-canned cmap or the one passed in by the user
    customcmap = False
    bv_cmaps = {'red_blue': [0.00, 0.6], 'purple_cyan': [0.8, 0.5], 'yellow_red': [0.00, 0.1667], 'red_green': [0.00, 0.3333],
                'blue_green': [0.333, 0.6], 'yellow_blue': [0.1667, 0.4667], 'pink_blue': [0.89, 0.55], 'orange_blue': [0.6, 0.1],
                'pink_orange': [0.89, 0.1]}
    try:
        try:
            xaxis, yaxis = bv_cmaps[cmap][0], bv_cmaps[cmap][1]
            xsat, ysat = 1, 1
        except:
            xcmap, ycmap = get_cmap(cmap[0]), get_cmap(cmap[1])
            customcmap = True
    except:
        try:
            xaxis, yaxis = cmap[0][0], cmap[1][0]
            if (mode == 'value'):
                xsat, ysat = cmap[0][1], cmap[1][1]
            elif (mode == 'saturation'):
                xsat, ysat = cmap[0][2], cmap[1][2]
            elif (mode == 'both'):
                xsat, ysat = cmap[0][2], cmap[1][2]
        except:
            raise Exception("Invalid colormap")

    # 10 Bin limit or an exception is raised
    if (d1num_bins > 10 or d2num_bins > 10):
        raise Exception("Bin number exceeds the 10 bin limit")

    colors = []
    d1normalized = d1norm(data1)
    d2normalized = d2norm(data2)
    if (d1maximum is None):
        d1max = np.amax(d1normalized)
    else:
        d1max = d1norm(d1maximum)
    if (d1minimum is None):
        d1min = np.amin(d1normalized)
    else:
        d1min = d1norm(d1minimum)
    if (d2maximum is None):
        d2max = np.amax(d2normalized)
    else:
        d2max = d2norm(d2maximum)
    if (d2minimum is None):
        d2min = np.amin(d2normalized)
    else:
        d2min = d2norm(d2minimum)

    if (d1bins is not None and isinstance(d1bins, (tuple, list, set))):
        d1binsnorm = d1norm(d1bins)
        d1num_bins = d1binsnorm.size - 1
        d1min = np.amin(d1binsnorm)
        d1max = np.amax(d1binsnorm)
    else:
        d1binsnorm = None
    if (d2bins is not None and isinstance(d2bins, (tuple, list, set))):
        d2binsnorm = d2norm(d2bins)
        d2num_bins = d2binsnorm.size - 1
        d2min = np.amin(d2binsnorm)
        d2max = np.amax(d2binsnorm)
    else:
        d2binsnorm = None

    c1range = d1max - d1min
    c2range = d2max - d2min
    # sjump and vjump is determined by the maxiumum bin number between the two bin numbers because if one is larger, the amount to
    # increase the saturation/value is automatically increased since at least one is increased
    sjump = 1.0 - (float(max(d1num_bins, d2num_bins)) / 11)
    vjump = float(max(d1num_bins, d2num_bins)) / 11

    if (d1extendfrac is None):
        d1extendfrac = 0.05
    if (d2extendfrac is None):
        d2extendfrac = 0.05
    # normalize each column using the normalizer passed in (default is linear)
    for i, j in zip(d1normalized, d2normalized):
        # The color value is obtained with the new _get_val function, which take extensions, custom limits, and bins into account.
        # See the _get_val function below with more details
        x = _get_val(d1extendfrac, d1extend, d1max, d1min,
                     d1num_bins, i, data1, d1binsnorm)
        # print(str(i) + " " + str(x))
        y = _get_val(d2extendfrac, d2extend, d2max, d2min,
                     d2num_bins, j, data2, d2binsnorm)
        # use the normalized data value to determine value of color & average them
        if (customcmap):
            r = _average_colors(matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(xcmap(x))),
                                matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(ycmap(y))), mode, darkenWhite=darkenWhite)
        elif (mode == 'value'):
            r = _average_colors([xaxis, xsat, x], [
                                yaxis, ysat, y], mode, vjump=vjump)
        elif (mode == 'saturation'):
            r = _average_colors([xaxis, x, xsat], [
                                yaxis, y, ysat], mode, sjump=sjump, darkenWhite=darkenWhite)
        elif (mode == 'both'):
            r = _average_colors([xaxis, x, xsat], [
                                yaxis, y, ysat], mode, sjump=sjump, vjump=vjump, darkenWhite=darkenWhite)
        colors.append(r)
    result = np.array(colors)

    # make the legend image (does not affect the mapping, it's just a legend for reference)

    # If there are bins are there are extensions, add to the number of bins according to the indicated extension
    if (d1num_bins > 1 and d1extend == "both"):
        d1num_bins = d1num_bins + 2
    if (d1num_bins > 1 and (d1extend == "min" or d1extend == "max")):
        d1num_bins = d1num_bins + 1
    if (d2num_bins > 1 and d2extend == "both"):
        d2num_bins = d2num_bins + 2
    if (d2num_bins > 1 and (d2extend == "min" or d2extend == "max")):
        d2num_bins = d2num_bins + 1
    first = []
    second = []
    spread = np.linspace(0, 1)
    for i in range(len(spread)):  # add 50 (default) evenly spaced colors of each color
        if (d1num_bins > 1):
            x = math.trunc(spread[i] * d1num_bins) / d1num_bins
        if (d2num_bins > 1):
            y = math.trunc(spread[i] * d2num_bins) / d2num_bins
        else:
            x = spread[i]
            y = spread[i]
        if (customcmap):
            first.append(matplotlib.colors.rgb_to_hsv(
                matplotlib.colors.to_rgb(xcmap(x))))
            second.append(matplotlib.colors.rgb_to_hsv(
                matplotlib.colors.to_rgb(ycmap(y))))
        elif (mode == 'saturation'):
            first.append((xaxis, x, xsat))
            second.append((yaxis, y, ysat))
        elif (mode == 'both'):
            first.append((xaxis, x, xsat))
            second.append((yaxis, y, ysat))
        elif (mode == 'value'):
            first.append((xaxis, xsat, x))
            second.append((yaxis, ysat, y))

    # my version of np.meshGrid(), but making sure to preserve the hsv color structure
    firstGrid = []
    secondGrid = []
    for i in first:
        firstGrid.append(first)
    for i in second:
        temp = []
        for j in second:
            temp.append(i)
        secondGrid.append(temp)

    # my version of np.dstack(), but instead of adding
    # I average the two colors and add the result to the Legend
    Legend = []
    for i, j in zip(firstGrid, secondGrid):
        temp = []
        for fcolor, scolor in zip(i, j):
            color = _average_colors(
                fcolor, scolor, mode, sjump=sjump, vjump=vjump, darkenWhite=darkenWhite)
            temp.append(color)
        Legend.append(temp)

    cax = ax
    if 'origin' not in imshow_kwargs and 'extent' not in imshow_kwargs:
        cax.imshow(Legend, origin="lower", extent=[
                   0, 1, 0, 1], **imshow_kwargs)
    if 'origin' in imshow_kwargs and 'extent' not in imshow_kwargs:
        cax.imshow(Legend, extent=[0, 1, 0, 1], **imshow_kwargs)
    if 'origin' not in imshow_kwargs and 'extent' in imshow_kwargs:
        cax.imshow(Legend, origin="lower", **imshow_kwargs)

    # set the new location of the legend if the user has requested a change
    if (legend_loc is not None):
        cax.set_position(legend_loc)
        # try: cax.set_position(legend_loc)
       # except: raise Exception("Invalid location parameter")

    # style the legend image
    # axis labels
    if (d1label == None):
        try:
            cax.set_xlabel(data1.name)
        except AttributeError:
            cax.set_xlabel("Variable 1")
    else:
        cax.set_xlabel(d1label)
    if (d2label == None):
        try:
            cax.set_ylabel(data2.name)
        except AttributeError:
            cax.set_ylabel("Variable 2")
    else:
        cax.set_ylabel(d2label)

    # tick labels (taking normalization, category labels, and extends into account)
    if (d1ticks is not None):
        cax.set_xticks(d1ticks)
    else:
        # Helper function to get the spacing and other information about where the ticks will be put
        # based on if there are any extensions, categories, or bins
        min1, max1, num1 = _get_tickinfo(
            d1num_bins, d1extend, d1extendfrac, d1category_labels)
        if (d1category_labels is None):
            cax.set_xticks(np.linspace(min1, max1, num=num1))
        else:
            cax.set_xticks(np.linspace(1 / (2 * d1num_bins),
                           1 - (1 / (2 * d1num_bins)), num=num1))
        if (d1category_labels is None):
            if (d1bins is None):
                xlabel = np.linspace(d1min, d2max, num=num1)
            else:
                xlabel = d1binsnorm
            for i in range(len(xlabel)):
                if (d1tick_rotation is None):
                    d1tick_rotation = 0  # Default tick rotations
                temp = (d1norm.inverse(xlabel[i]) - d1norm.inverse(xlabel[0])) / (d1norm.inverse(xlabel[len(xlabel) - 1])
                                                                                  - d1norm.inverse(xlabel[0]))
                temp = temp * (d1norm.inverse(d1max) -
                               d1norm.inverse(d1min)) + d1norm.inverse(d1min)
                xlabel[i] = round(temp, 1)
            # making sure to apply the apporopriate tick rotations
            cax.set_xticklabels(xlabel, rotation=d1tick_rotation)
        else:
            xlabels = []
            if (d1tick_rotation is None):
                d1tick_rotation = 30
            for i in range(d1num_bins):
                xlabels.append(d1category_labels[i % len(d1category_labels)])
            # make sure to rotate so that the labels aren't jumbled together
            cax.set_xticklabels(xlabels, rotation=d1tick_rotation)
    if (d2ticks is not None):
        cax.set_yticks(d2ticks)
    else:
        min2, max2, num2 = _get_tickinfo(
            d2num_bins, d2extend, d2extendfrac, d2category_labels)
        if (d2category_labels == None):
            cax.set_yticks(np.linspace(min2, max2, num=num2))
        else:
            cax.set_yticks(np.linspace(1 / (2 * d2num_bins),
                           1 - (1 / (2 * d2num_bins)), num=num2))
        if (d2category_labels is None):
            if (d2bins is None):
                ylabel = np.linspace(d2min, d2max, num2)
            else:
                ylabel = d2binsnorm
            for i in range(len(ylabel)):
                if (d2tick_rotation is None):
                    d2tick_rotation = 0
                temp = (d2norm.inverse(ylabel[i]) - d2norm.inverse(ylabel[0])) / (d2norm.inverse(ylabel[len(ylabel) - 1])
                                                                                  - d2norm.inverse(ylabel[0]))
                temp = temp * (d2norm.inverse(d2max) -
                               d2norm.inverse(d2min)) + d1norm.inverse(d2min)
                ylabel[i] = round(temp, 2)
            cax.set_yticklabels(ylabel, rotation=d2tick_rotation)
        else:
            ylabels = []
            if (d2tick_rotation is None):
                d2tick_rotation = 30
            for i in range(d2num_bins):
                ylabels.append(d2category_labels[i % len(d2category_labels)])
            # make sure to rotate so that the labels aren't jumbled together
            cax.set_yticklabels(ylabels, rotation=d2tick_rotation)

    # apply custom font properties to the axis labels if applicable
    if isinstance(label_fontproperties, matplotlib.font_manager.FontManager):
        cax.set_yticklabels(cax.get_yticklabels,
                            fontproperties=label_fontproperties)
        cax.set_xticklabels(cax.get_xticklabels,
                            fontproperties=label_fontproperties)

    # title and its font properties
    if (title == None):
        cax.set_title("Legend")
    else:
        cax.set_title("" + title)
    if (title_fontproperties == None):
        cax.set_title(cax.get_title(), fontsize=10)
    elif isinstance(title_fontproperties, matplotlib.font_manager.FontManager):
        cax.set_title(cax.get_title(),
                      fontproperties=title_fontproperties)

    # return the result
    return result, cax

# takes in two HSV colors to blend and the blending mode (value or saturation)


def _average_colors(fcolor, scolor, mode, sjump=.5, vjump=.1, darkenWhite=False):
    f = matplotlib.colors.hsv_to_rgb(fcolor)  # convert the hsv to rgb colors
    s = matplotlib.colors.hsv_to_rgb(scolor)
    # take the square root of the average of the squares of the red, green, and blue components
    r = (math.sqrt((f[0] * f[0] + s[0] * s[0]) / 2), math.sqrt((f[1]
         * f[1] + s[1] * s[1]) / 2), math.sqrt((f[2] * f[2] + s[2] * s[2]) / 2))

    # convert the result color back to hsv to increase the brightness
    to_hsv = matplotlib.colors.rgb_to_hsv(r)
    if (mode == 'value'):
        # 1 + sjump and vjump because otherwise the saturation/value is decreased by vjump or sjump (like 50%) instead of being increased
        # with the pervious version, I was getting very dark colors/legends, especially when the number of bins went up
        to_hsv[2] = min(1, (to_hsv[2] * (1 + vjump)))
    if (mode == 'saturation'):
        to_hsv[1] = min(1, (to_hsv[1] * (1.5 + sjump)))
        # If the color is white and the user wants to darken the white color, it will be changed to a 25% grey color to pick out easier
        if (darkenWhite == True and np.array_equal(to_hsv, (to_hsv[0], 0, 1))):
            to_hsv = (to_hsv[0], 0, 0.75)
    if (mode == 'both'):
        to_hsv[1] = min(1, (to_hsv[1] * 1.4))
        to_hsv[2] = min(1, (to_hsv[2] * 0.9))
        if (darkenWhite == True and np.array_equal(to_hsv, (to_hsv[0], 0, 1))):
            to_hsv = (to_hsv[0], 0, 0.75)
    # convert the brightened color back to rgb and return this
    return matplotlib.colors.hsv_to_rgb(to_hsv)

# This is a helper method used to take a value from an originial range and scale it down to a new indicated range


def _scale_values(original_min, original_max, new_min, new_max, vals):
    if isinstance(vals, (tuple, list, set)):
        result = []
        for i in vals:
            r = ((i - original_min) / (original_max - original_min)
                 * (new_max - new_min)) + new_min
            result.append(r)
        return result
    else:
        return ((vals - original_min) / (original_max - original_min) * (new_max - new_min)) + new_min

# Helper method to get the color value of each data point


def _get_val(extendfrac, extend, max, min, num_bins, val, column, bins):
    if (extendfrac is None):
        extendfrac = 0.05  # default extension of a continuous map on both ends is 5%
    if (extend is None):
        # out-of-bound values are mapped to the highest/lowest possible value
        # in-bound values that are equal to the min/max can also be mapped to the highest/lowest value
        if (val > max):
            x = 1
        elif (val < min):
            x = 0
        else:
            x = _scale_values(min, max, 0, 1, val)
        if (num_bins > 1):
            # The below uses a new method to chop values into bins where each bin's range is determined by
            # [min_val + n*w] where w = (max-min)/(no of bins)

            # The bins have to be scaled to the proper color limits
            if (bins is None):
                bins_scaled = np.linspace(0, 1, num=(num_bins + 1))
            else:
                bins_scaled = _scale_values(
                    np.amin(bins), np.amax(bins), 0, 1, bins)
            num_bin = np.digitize(x, bins_scaled)
            # [0, 0.5, 1, 1.5, 2]
            # the above returns the index of the bin (which is the list of values from np.linspace) that the given value falls into
            w = 1 / num_bins
            x = (num_bin - 1) * w
    elif (extend == "both"):
        if (val > max):
            if (num_bins > 1):
                x = 1  # value of the highest bin
            # scale it on the fraction of the end of the colorscale reserved for values higher than the max
            else:
                x = _scale_values(max, np.amax(column),
                                  (1 - extendfrac), 1, val)
        elif (val < min):
            if (num_bins > 1):
                x = 0
            else:
                x = _scale_values(np.amin(column), min, 0, extendfrac, val)
        else:
            if (num_bins > 1):
                # scale to given number from the second bin the to second-to-last bin
                a = _scale_values(min, max, 1 / (num_bins + 2),
                                  (num_bins + 1) / (num_bins + 2), val)
                # this is the w referenced above in the new binning method used
                w = ((num_bins + 1) / (num_bins + 2) -
                     (1 / (num_bins + 2))) / num_bins
                if (bins is None):
                    bins_scaled = np.linspace(
                        (1 / (num_bins + 2)), (num_bins + 1) / (num_bins + 2), num=(num_bins + 1))
                else:
                    bins_scaled = _scale_values(np.amin(bins), np.amax(
                        bins), 1 / (num_bins + 2), (num_bins + 1) / (num_bins + 2), bins)
                num_bin = np.digitize(a, bins_scaled)
                x = 1 / (num_bins + 2) + ((num_bin - 1) * w)
            else:
                x = _scale_values(min, max, extendfrac, (1 - extendfrac), val)
    # min and max extends is similar to "both" extends, but only one end of the color scale is extended
    # (a fraction is taken out if continuous or a a bin is added and the first/last bin is used for out-of-bound values)
    elif (extend == "min"):
        if (val > max):
            x = 1
        if (val < min):
            if (num_bins > 1):
                x = 0
            else:
                x = _scale_values(min, max, 0, extendfrac, val)
        else:
            if (num_bins > 1):
                a = _scale_values(min, max, 1 / (num_bins + 1), 1, val)
                w = (1 - (1 / (num_bins + 1))) / num_bins
                if (bins is None):
                    bins_scaled = np.linspace(
                        (1 / (num_bins + 1)), 1, num=(num_bins + 1))
                else:
                    bins_scaled = _scale_values(
                        np.amin(bins), np.amax(bins), 1 / (num_bins + 1), 1, bins)
                num_bin = np.digitize(a, bins_scaled)
                x = 1 / (num_bins + 1) + ((num_bin - 1) * w)
            else:
                x = _scale_values(min, max, extendfrac, 1, val)
    elif (extend == "max"):
        if (val > max):
            if (num_bins > 1):
                a = _scale_values(
                    min, max, 0, (num_bins / (num_bins + 1)), val)
                w = (num_bins / (num_bins + 1)) / num_bins
                if (bins is None):
                    bins_scaled = np.linspace(
                        0, num_bins / (num_bins + 1), num=(num_bins + 1))
                else:
                    bins_scaled = _scale_values(np.amin(bins), np.amaxx(
                        bins), 0, num_bins / (num_bins + 1), bins)
                num_bin = np.digitize(a, bins_scaled)
                x = (num_bin - 1) * w
            else:
                x = _scale_values(min, max, (1 - extendfrac), 1, val)
        if (val < min):
            x = 0
        else:
            x = _scale_values(min, max, 0, (1 - extendfrac), val)
    return x

# helper function to get the minimum and maximum location of the ticks and how many ticks to place in between those locations
# takes into acount category labels, bins, and extensions


def _get_tickinfo(num_bins, extend, extendfrac, categories):
    if (num_bins == 1):
        num1 = 5  # five ticks if continuous by default
    elif (categories != None):
        num1 = num_bins  # if there are category labels, then each bin will get its own tick in the middle of it
    elif (extend == "both"):
        num1 = num_bins - 1
    elif (extend == "min" or extend == "max"):
        num1 = num_bins
    else:
        num1 = num_bins + 1
    if (num_bins > 1):
        if (extend == "both"):
            min1 = 1 / num_bins
            max1 = (num_bins - 1) / num_bins
        elif (extend == "min"):
            min1 = 1 / num_bins
            max1 = 1
        elif (extend == "max"):
            min1 = 0
            max1 = (num_bins - 1) / num_bins
        else:
            min1 = 0
            max1 = 1
    else:
        if (extend == "both"):
            min1 = extendfrac
            max1 = 1 - extendfrac
        elif (extend == "min"):
            min1 = extendfrac
            max1 = 1
        elif (extend == "max"):
            min1 = 0
            max1 = 1 - extendfrac
        else:
            min1 = 0
            max1 = 1
    return min1, max1, num1
