import matplotlib
matplotlib.use("QtAgg")   # ensure Qt backend is used

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot([1,2,3,4], [1,4,2,5])

plt.title("Overlay Test - Drag Mouse")

plt.show()