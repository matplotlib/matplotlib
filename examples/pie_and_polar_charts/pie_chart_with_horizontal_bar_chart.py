"""
Simple pie chart with basic features and horizontal bar chart in
subplot form.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.figure(figsize=(14,8))

#Basic chart data
c_items = [35, 15, 25, 8, 5, 42]
c_item_labels = ['London', 'Barcelona', 'Berlin', 'Paris', 'Amsterdam', 'Bern']
colors = ['#FF0000', '#00FF00', '#0055FF', '#FFFF00', '#CCEEFF','#FF00FF' ]

explode = [0, 0, 0, 0.15, 0.15, 0] #Explode values for pie chart

grid_1 = GridSpec(1,2) #Subplot in 1 row by 2 columns

#Horizontal Bar Chart
plt.subplot(grid_1[0, 0], aspect=6)
y_axis_pos=np.arange(len(c_item_labels))
plt.barh(y_axis_pos, c_items, color=colors, align='center', alpha=1)
plt.yticks(y_axis_pos,c_item_labels)

#Pie Chart
plt.subplot(grid_1[0, 1], aspect=1)
plt.pie(c_items, labels=c_item_labels, colors=colors, explode=explode, autopct='%1.1f%%')
plt.axis('equal')

plt.show()
