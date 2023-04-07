import matplotlib.pyplot as plt

x = ['Mapel', 'Hazel', 'Willow', 'Birch']
y = [10, 13, 5, 16]

fig, ax = plt.subplots()
ax.bar(x, y)

ax.label_by_line(fig, x, static_pos=None)

plt.show()
