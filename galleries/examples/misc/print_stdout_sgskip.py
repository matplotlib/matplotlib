"""
=====================
Print image to stdout
=====================

print png to standard out

usage: python print_stdout.py > somefile.png

"""

import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_path = '_static/print_stdout_thumbnail.svg'

plt.plot([1, 2, 3])
plt.savefig(sys.stdout.buffer)
