import matplotlib.pyplot as plt

x = ['T1', 'T2', 'T3', 'T4']
y = [10, 13, 5, 16]
names = ['Mapel', 'Hazel', 'Willow', 'Birch']

fig, ax = plt.subplots()
ax.bar(x, y)

ax.label_by_data(names, fig)

plt.show()
