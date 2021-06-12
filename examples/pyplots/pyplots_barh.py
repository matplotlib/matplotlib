import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.broken_barh([(120, 50), (160, 10)], (20, 12), facecolors='tab:blue')
ax.broken_barh([(30, 70), (150, 20), (170, 30)], (10, 9),
               facecolors=('tab:orange', 'tab:green', 'tab:red'))
ax.set_ylim(5, 35)
ax.set_xlim(0, 400)
ax.set_xlabel('seconds')
ax.set_yticks([15, 25])
ax.set_yticklabels(['Bill', 'Jim'])
ax.grid(True)

plt.show()
