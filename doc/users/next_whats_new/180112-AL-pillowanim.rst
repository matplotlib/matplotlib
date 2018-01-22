Writing animations with Pillow
``````````````````````````````
It is now possible to use Pillow as an animation writer.  Supported output
formats are currently gif (Pillow>=3.4) and webp (Pillow>=5.0).  Use e.g. as ::

   from __future__ import division

   from matplotlib import pyplot as plt
   from matplotlib.animation import FuncAnimation, PillowWriter

   fig, ax = plt.subplots()
   line, = plt.plot([0, 1])

   def animate(i):
      line.set_ydata([0, i / 20])
      return [line]

   anim = FuncAnimation(fig, animate, 20, blit=True)
   anim.save("movie.gif", writer=PillowWriter(fps=24))
   plt.show()
