"""
You can use matplotlib to generate thumbnails from existing images.
matplotlib natively supports PNG files on the input side, and other
image types transparently if your have PIL installed
"""

from __future__ import print_function
# build thumbnails of all images in a directory
import sys, os, glob
import matplotlib.image as image


if len(sys.argv)!=2:
    print('Usage: python %s IMAGEDIR'%__file__)
    raise SystemExit
indir = sys.argv[1]
if not os.path.isdir(indir):
    print('Could not find input directory "%s"'%indir)
    raise SystemExit

outdir = 'thumbs'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for fname in glob.glob(os.path.join(indir, '*.png')):
    basedir, basename = os.path.split(fname)
    outfile = os.path.join(outdir, basename)
    fig = image.thumbnail(fname, outfile, scale=0.15)
    print('saved thumbnail of %s to %s'%(fname, outfile))

