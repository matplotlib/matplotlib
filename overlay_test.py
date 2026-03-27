import matplotlib.pyplot as plt

fig, ax = plt.subplots()
canvas = fig.canvas

plt.show(block=False)

canvas.add_overlay_line(50, 50, 300, 300)
canvas.add_overlay_line(50, 300, 300, 50)

canvas.draw()
plt.pause(0.1)
