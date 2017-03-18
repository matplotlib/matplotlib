from matplotlib import pyplot as plt

from matplotlib import colors, cm, table

import random

def gendata(n=5, k=8):


    data = []
    for value in range(n):
        a = random.random()
        b = random.random()
        values = [random.gauss(a * 10**4, b * 10**4) for x in range(k)]

        data.append(values)

    return data

def draw_table():
    """ Draw a table on the axes """

    norm = colors.Normalize()

    data = gendata()
    data.sort(key=sum)

    rows = len(data)
    cols = len(data[0])

    axes = plt.axes()

    rcolours = []
    for row in data:
        row.sort()
        patches = axes.plot(row)
        rcolours.append(patches[0].get_color())


    rcolours = [colors.to_rgba(x, 0.2) for x in rcolours]

    means = [sum(x) / len(x) for x in data]
    rlabels = ['Mean {x:,.0f}'.format(x=x) for x in means]
    
    colours = cm.get_cmap()(norm(data))
    alpha = 0.2
    colours[:, :, 3] = alpha

    text = []
    for row in data:
        rtext = ['{x:,.0f}'.format(x=x) for x in row]
        text.append(rtext)

    plt.subplots_adjust(left=0.2, bottom=0.3)

    tab = axes.table(
        rowLabels = rlabels,
        rowColours = rcolours,
        rowEdgeColours = rcolours,
        cellText = text,
        cellColours = colours,
        cellEdgeColours=colours,
        loc='bottom')

    axes.set_axis_off()

    plt.title('Shaded table')


draw_table()
plt.show()
    
    
