"""
==============
Frame grabbing
==============

Use a MovieWriter directly to grab individual frames and write them to a
file.  This avoids any event loop integration, and thus works even with the Agg
backend.  This is not recommended for use in an interactive setting.
"""

# sphinx_gallery_thumbnail_path = "_static/frame_grabbing.png"
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

# Fixing random state for reproducibility
np.random.seed(19680801)


metadata = dict(title='Movie Test', artist='Matplotlib')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig, ax = plt.subplots()
l0, = ax.plot([], [], color='C0', label='moving line')
l1, = ax.plot([], [], color='C1', label='moving circle')
ax.legend()
ax.set(xlim=[-1, 1], ylim=[-1, 1])

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        l0.set_data(np.linspace(-1, 1, 100),
                    np.sin(2 * np.pi * (i / 100) + np.linspace(-1, 1, 100) * 2 * np.pi))
        l1.set_data(np.cos(2 * np.pi * i / 100), np.sin(2 * np.pi * i / 100))
        writer.grab_frame()
