"""
==================================
Using the full command line option
==================================

This example show:
- how to get a slow animation work in for example the VLC player
- how to use the full command line option when the default command line is not
  general enough
- one way to animate an axes title with `.ArtistAnimation`
- several ways of saving the animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.close('animation')
fig, ax = plt.subplots(num='animation')

# There is a problem with animation of the axis title with artist animation.
# The problem is because the title is one object in the axes object that just
# get updated and not changed. This solution use a an ordinary text object with
# some of the title properties so that it behaves like the axes title.
transform = ax.title.get_transform()
title_kwargs = dict(transform=transform, horizontalalignment='center')

def f(x, y, i):
    return np.sin(x+i*np.pi/15) + np.cos(y+i*np.pi/20)

x = np.linspace(0, 2*np.pi, 120)
y = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)

ims = []
for i in range(10):
    im = ax.imshow(f(x, y, i))
    number = ax.text(60, 50, str(i), fontsize=50, ha='center',
                     transform=ax.transData)
    title = ax.text(0.5, 1, 'Frame number: {}'.format(i), **title_kwargs)
    ims.append([im, number, title])

anim = animation.ArtistAnimation(fig, ims, interval=2000, repeat_delay=1000)

# The animation can be saved to an animated gif with the ImageMagick or Pillow
# writers with for example.
anim.save('anim.gif', writer='imagemagick')

# or to a html file with for example.
writer_html = animation.HTMLWriter(fps=0.5, embed_frames=True)
anim.save('anim.html', writer=writer_html)

# It can also be saved to an .mp4 file with the ffmpeg or avconv writers with
# for example.

writer1 = animation.FFMpegFileWriter(fps=0.5)
anim.save('anim1.mp4', writer=writer1)

##########################
# ``anim1.mp4`` might not work well in all movie players though. The proplems is
# that the frame rate of the movie is 0.5 frames per second and all players cant
# handle such a low frame rate. This can be solved by having different input
# and output frame rates. You can read more here_.

# This can be done in matplotlib using the full command line option.

# .. _here: rac.ffmpeg.org/wiki/Slideshow

# See how the command used before looks like.
command1 = writer1.get_command()
print(command1)
# ffmpeg -r 0.5 -i _tmp%07d.png -vframes 11 -vcodec h264 -pix_fmt
# yuv420p -y anim1.mp4
#
# We can change that command line to a formated one to be able to save the
# movie with an output framerate of 25 fps.
full_command = ['{path} -framerate {fps} -i _tmp%07d.png  -vf fps=25 -vcodec '
                'h264 -pix_fmt yuv420p -y {outfile}'][0]

writer2 = animation.FFMpegFileWriter(fps=0.5, extra_args=full_command)
anim.save('anim2.mp4', writer=writer2)

# Another way to achieve a workable movie is the use the extra_args option
# in the FFMpegWriter (the same method doesn't work in the FFMpegFileWriter
# due to the -vframes argument).
writer3 = animation.FFMpegWriter(fps=0.5, extra_args=['-vf', 'fps=25'])
anim.save('anim3.mp4', writer=writer3)

plt.show()

##############################
# References
# ----------
#
# The use of the following functions and methods is shown
# in this example:

import matplotlib
matplotlib.animation.ArtistAnimation
matplotlib.animation.ArtistAnimation.save
matplotlib.animation.ImageMagickWriter
matplotlib.animation.HTMLWriter
matplotlib.animation.FFMpegFileWriter
matplotlib.animation.FFMpegFileWriter.get_command
