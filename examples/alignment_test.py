"""
You can precisely layout text in data or axes (0,1) coordinates.  This
example shows you some of the alignment and rotation specifications to
layout text
"""

from matplotlib.matlab import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Transform

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ax = gca()
p = Rectangle(ax.dpi, ax.bbox,
              (left, bottom), width, height,
              fill=False,
              transx=ax.xaxis.transAxis,
              transy=ax.yaxis.transAxis)
p.set_clip_on(False)
ax.add_patch(p)


ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(left, 0.5*(bottom+top), 'leftcenter',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transx=ax.xaxis.transAxis,
        transy=ax.yaxis.transAxis)

axis('off')
#savefig('alignment_test', dpi=300)
show()
