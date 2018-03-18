import matplotlib.pyplot as plt

x = [0.5, -1.0, 1.0]
y = [1.0, -2.0, 3.0]
data = ['100.1', 'Hello!', '50']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scattertext(x, y, data, loc=(-20, 20))
ax.scattertext(x, y, data, color='red', loc=(20, -20))
ax.scattertext(x, y, data, color='blue')

plt.show()
