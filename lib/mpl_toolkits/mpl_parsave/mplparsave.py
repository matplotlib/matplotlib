## Record an MPL movie in parallel using multiple threads.
import os
from matplotlib.animation import FuncAnimation as fanim
from multiprocessing import Process
import subprocess
import time

# === PARALLEL SAVE CLASS ================================================#
# Speed up writing matplotlib animations to a movie file by using multiple
# processes.
# ========================================================================#
class Parsave:
  def __init__(self):
    pass

  # Record the movie. The parameters are
  #
  # fname: output file name.
  #
  # init: the function that draws background of each frame.
  #
  # fig: figure where animation will be drawn.
  #
  # anim_func: the function to be animated.
  #
  # blocks: an array of the form [block_1, block_2, ..., block_n]
  # where for j = 1, ..., n, block_j is the (array of) frames to be recorded
  # by the jth thread using writerClass, which at the end is stitched together
  # into a single movie using stitcherClass.
  #
  # writerClass: the writer responsible for writing the frames
  # (an instance of the matplotlib.animation.MovieWriter).
  #
  # stitcherClass: the stitcher responsible for stitching the movies recorded
  # by the writerClass into a single movie.
  #
  # The number of threads that will be run is equal to n
  # (the number of blocks as above).
  #
  # keep: True or False; whether to keep the parts that are then stitched
  # into the final movie, or delete them after stitching the final movie.
  #
  # **kwargs are the keyword arguments to be passed to anim_func.
  @staticmethod
  def record(fname, fig, anim_func, init_func, blocks,
             writerClass, stitcherClass, keep=False, **kwargs):
    num_jobs=len(blocks)
    jobs=[0]*num_jobs
    names=[0]*num_jobs

    # Record movies.
    for j in range(num_jobs):
      names[j]=(('%d'+'-'+'%f'+'.'+fname.split('.')[-1])) % \
                (j, float(time.time()))
      jobs[j]=Process(target=Parsave.__parallel_save,
                      args=(names[j], fig, init_func, anim_func, blocks[j],
                            writerClass), kwargs=kwargs)
      jobs[j].start()

    for job in jobs:
      job.join()

    # Now stitch the recorded movies together.
    if num_jobs==1:
      os.rename(names[0], fname)
    else:
      stitcherClass.stitch(fname, names)
      if keep is False:
        for name in names:
          os.remove(name)

  # The helpber for record(). 'frames' is a block from 'blocks' passed to
  # record().
  @staticmethod
  def __parallel_save(fname, fig, init, anim_func,
                      frames, writerClass, **kwargs):
    anm=fanim(fig, anim_func, init_func=init, frames=frames, **kwargs)
    anm.save(fname, writer=writerClass)
# ========================================================================#







# === STITCHER WRAPPER ===================================================#
# Wrapper for the stitcher classes
# ========================================================================#
supported_stitchers=['ffmpeg',
                     'mencoder']

class Stitcher:
  # name: the name of the class to use, e.g. 'mencoder'.
  #
  # args: arguments for the stitcher.
  def __init__(self, name, args=None):
    if name=='ffmpeg':
      self.stitcher=stitcher_ffmpeg(args)
    elif name=='mencoder':
      self.stitcher=stitcher_mencoder(args)
    else:
      error='Unsupported stitcher. Supported stitchers: '
      for s in supported_stitchers:
        error+=(s+', ')
      error=error[:-2]+'.'
      raise ValueError(error)

  # fname: the output file name.
  # fnames = [fname_1, fname_2, ..., fname_n] the file names of the movies
  # to stitch.
  def stitch(self, fname, fnames):
    self.stitcher.stitch(fname, fnames)
# ========================================================================#







# === STITCHERS ==========================================================#
# Currently supported:
#  - mencoder
#  - ffmpeg
#
#
# === MENCODER STITCHER ==================================================#
# Stitcher based on 'mencoder'.
# ========================================================================#
class stitcher_mencoder:
  def __init__(self, args=None):
    self.args=args

  def stitch(self, fname, fnames):
    if self.args is None:
      self.args=['-ovc', 'copy', '-idx', '-o']+[fname]+fnames

    subprocess.check_call(['mencoder']+self.args,
                          stdout=open(os.devnull, 'w'),
                          stderr=subprocess.STDOUT)
# ========================================================================#







# === FFMPEG STITCHER ====================================================#
# Stitcher based on 'ffmpeg'.
# ========================================================================#
class stitcher_ffmpeg:
  def __init__(self, args=None):
    self.args=args

  def stitch(self, fname, fnames):
    input_file=(('%s%f'+'.txt')%('input', float(time.time())))
    f=open(input_file, 'w')
    for name in fnames:
      f.write('file '+"'"+name+"'"+'\n')
    f.close()
    args=['-f', 'concat', '-i', input_file, '-codec', 'copy']+[fname]
    if self.args is not None:
      args=self.args+args

    subprocess.check_call(['ffmpeg']+args,
                          stdout=open(os.devnull, 'w'),
                          stderr=subprocess.STDOUT)
    os.remove(input_file)
# ========================================================================#
