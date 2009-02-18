#=======================================================================
""" A set of utilities for manipulating path information.
"""
#=======================================================================

import os
import shutil
import os.path

#=======================================================================

__all__ = [
            'chdir',
            'exists',
            'extension',
            'joinPath',
            'mkdir',
            'name',
            'rm',
            'rmdir',
            'walk',
          ]

#-----------------------------------------------------------------------
def chdir( path ):
   """Change the current working directory to the specified directory."""
   os.chdir( path )

#-----------------------------------------------------------------------
def exists( path ):
   """Returns true if the specified path exists."""
   return os.path.exists( path )

#-----------------------------------------------------------------------
def extension( path ):
   """Returns the extension name of a filename."""
   unused, ext = os.path.splitext( path )
   return ext

#-----------------------------------------------------------------------
def joinPath( *args ):
   """Returns true if the specified path exists."""
   return os.path.join( *args )

#-----------------------------------------------------------------------
def mkdir( path, mode = 0777, recursive = False ):
   """Create the specified directory."""
   if recursive:
      os.makedirs( path, mode )
   else:
      os.mkdir( path, mode )

#-----------------------------------------------------------------------
def name( path ):
   """Returns the name portion of a specified path."""
   return os.path.basename( path )

#-----------------------------------------------------------------------
def rm( path ):
   """Remove the specified file."""
   os.remove( path )

#-----------------------------------------------------------------------
def rmdir( path ):
   """Remove the specified directory."""
   shutil.rmtree( path, ignore_errors = True )

#-----------------------------------------------------------------------
def walk( path ):
   """Recursively iterate over files and sub-directories."""
   children = os.listdir( path )
   children = [ os.path.join( path, child ) for child in children ]

   for child in children:
      yield child

      if os.path.isdir( child ):
         for grandchild in walk( child ):
            yield grandchild


