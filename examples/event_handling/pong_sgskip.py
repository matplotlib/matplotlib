"""
====
Pong
====

A small game demo using Matplotlib.

.. only:: builder_html

   This example requires :download:`pipong.py <pipong.py>`

"""
import time


import matplotlib.pyplot as plt
import pipong


fig, ax = plt.subplots()
canvas = ax.figure.canvas
animation = pipong.Game(ax)

# disable the default key bindings
if fig.canvas.manager.key_press_handler_id is not None:
    canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)


# reset the blitting background on redraw
def on_redraw(event):
    animation.background = None


# bootstrap after the first draw
def start_anim(event):
    canvas.mpl_disconnect(start_anim.cid)

    def local_draw():
        if animation.ax.get_renderer_cache():
            animation.draw(None)
    start_anim.timer.add_callback(local_draw)
    start_anim.timer.start()
    canvas.mpl_connect('draw_event', on_redraw)


start_anim.cid = canvas.mpl_connect('draw_event', start_anim)
start_anim.timer = animation.canvas.new_timer(interval=1)

tstart = time.time()

plt.show()
print('FPS: %f' % (animation.cnt/(time.time() - tstart)))
