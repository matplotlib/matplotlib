from distutils.core import setup
import sys,os


setup(name="matplotlib",
      version= '0.2b',
      description = "Matlab style plotting lib using pygtk",
      author = "John D. Hunter",
      author_email="jdhunter@ace.bsd.uchicago.edu",
      url = "http://nitace.bsd.uchicago.edu:8080/matplotlib",
      long_description = """
      matplotlib strives to produce publication quality 2D graphics
      using matlab plotting for inspiration.  Although the main lib is
      object oriented, there is a functional matlab style interface
      for people coming from matlab.
      """,
      packages=['matplotlib'],
      platforms='any',
      )
