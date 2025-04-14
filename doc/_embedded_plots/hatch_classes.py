import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

pattern_to_class = {
    '/': 'NorthEastHatch',
    '\\': 'SouthEastHatch',
    '|': 'VerticalHatch',
    '-': 'HorizontalHatch',
    '+': 'VerticalHatch + HorizontalHatch',
    'x': 'NorthEastHatch + SouthEastHatch',
    'o': 'SmallCircles',
    'O': 'LargeCircles',
    '.': 'SmallFilledCircles',
    '*': 'Stars',
}

for i, (hatch, classes) in enumerate(pattern_to_class.items()):
    r = Rectangle((0.1, i+0.5), 0.8, 0.8, fill=False, hatch=hatch*2)
    ax.add_patch(r)
    h = ax.annotate(f"'{hatch}'", xy=(1.2, .5), xycoords=r,
                    family='monospace', va='center', ha='left')
    ax.annotate(pattern_to_class[hatch], xy=(1.5, .5), xycoords=h,
                family='monospace', va='center', ha='left', color='tab:blue')

ax.set(xlim=(0, 5), ylim=(0, i+1.5), yinverted=True)
ax.set_axis_off()
