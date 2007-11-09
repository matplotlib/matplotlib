#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#
# Author: David C. Morrill
# Date: 11/20/2004
#------------------------------------------------------------------------------
""" Defines a database for traits.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import shelve
import inspect
import atexit
import os

from os.path      import split, splitext, join, exists
from traits       import CTrait, Property, Str, Dict, true, false, trait_from
from has_traits   import HasPrivateTraits
from trait_base   import SequenceTypes, traits_home
from trait_errors import TraitError

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Name of a traits data base file
DB_NAME = '__traits__'

#-------------------------------------------------------------------------------
#  'TraitDB' class:
#-------------------------------------------------------------------------------

class TraitDB ( HasPrivateTraits ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Default package name
    default       = Str( 'global', minlen = 1 )

    # Name of the Trait DB
    file_name     = Str

    # Is current Trait DB read only?
    read_only     = true

    #rdb          = Trait( shelve_db ) # Current underlying 'shelve' data base
    #wdb          = Trait( shelve_db ) # Writable version of current 'shelve' db

    # Map of { directory: package_name }
    _package_map  = Dict

    # Has 'atexit' call been registered yet?
    _at_exit      = false

    # Is a 'batch update' in progress?
    _batch_update = false

    #---------------------------------------------------------------------------
    #  Defines the 'rdb' and 'wdb' properties:
    #---------------------------------------------------------------------------

    def _get_rdb ( self ):
        if self._db is None:
            file_name = self._file_name()
            try:
                self._db = shelve.open( file_name, flag = 'r', protocol = -1 )
            except:
                self._db = shelve.open( file_name, flag = 'c', protocol = -1 )
                self._init_db()
                self.update()
                self._db = shelve.open( file_name, flag = 'r', protocol = -1 )
            self.read_only = True
        return self._db

    def _get_wdb ( self ):
        if self._db is not None:
            if not self.read_only:
                return self._db
            self._db.close()
            self._db = None
        self._db = db = shelve.open( self._file_name(), flag      = 'c',
                                                        protocol  = -1,
                                                        writeback = True )
        self.read_only = False
        if db.get( '@names' ) is None:
            self._init_db()
        return db

    def _set_db ( self, db ):
        if self._db is not None:
            self._db.close()
            self._db = None

    rdb = Property( _get_rdb, _set_db )
    wdb = Property( _get_wdb, _set_db )

    #---------------------------------------------------------------------------
    #  Initializes a new underlying 'shelve' data base:
    #---------------------------------------------------------------------------

    def _init_db ( self ):
        db = self._db
        db[ '@names' ]      = []
        db[ '@dbs' ]        = {}
        db[ '@children' ]   = {}
        db[ '@categories' ] = {}

    #---------------------------------------------------------------------------
    #  Gets/Sets the definition of a trait in the current data base:
    #---------------------------------------------------------------------------

    def __call__ ( self, name, trait = -1 ):
        """ Gets/Sets the definition of a trait in the current data base.
        """
        if trait == -1:
            rdb = self.rdb

            # If the package name was explictly specified, just look it up:
            if name.find( '.' ) >= 0:
                try:
                    return rdb[ name ]
                except:
                    pass
            else:
                # Otherwise, see if it exists using the default package:
                try:
                    return rdb[ '%s.%s' % ( self.default, name ) ]
                except:
                    # Otherwise see if it is unique across all packages, and
                    # return the unique value if it is:
                    try:
                        packages = rdb[ name ]
                        if len( packages ) == 1:
                            return rdb[ '%s.%s' % ( packages[0], name ) ]
                    except:
                        pass

            # Couldn't find a trait definition, give up:
            raise ValueError, 'No trait definition found for: ' + name

        # Make sure that only valid traits are stored in the data base:
        if trait is not None:
            trait = trait_from( trait )

        col = name.rfind( '.' )
        if col < 0:
            # If the name does not include an explicit package, use the default
            # package:
            package   = self.default
            base_name = name
            name      = '%s.%s' % ( package, base_name )
        else:
            # Else remember the base name and explicit package name specified:
            base_name = name[col+1:]
            package   = name[:col]

        # Get a writable data base reference:
        db = self.wdb

        # If there was a previous definition for the trait, remove it:
        db_trait = db.get( name )
        if db_trait is not None:
            del db[ name ]
            db[ base_name ].remove( package )
            db[ '@names' ].remove( name )
            parent = db_trait.parent
            if isinstance(parent, basestring):
                parent = self._package_name( parent )
                db[ '@children' ][ parent ].remove( name )
            categories = db_trait.categories
            if isinstance(categories, basestring):
                categories = [ categories ]
            if type( categories ) in SequenceTypes:
                db_categories = db[ '@categories' ]
                for category in categories:
                    db_categories[ category ].remove( name )

        # Define the new trait (if one was specified):
        if trait is not None:
            db[ name ] = trait
            db.setdefault( base_name, [] )
            db[ base_name ].append( package )
            db[ '@names' ].append( name )
            db[ '@names' ].sort()
            parent = trait.parent
            if isinstance(parent , basestring):
                parent = self._package_name( parent )
                db[ '@children' ][ parent ].append( name )
            categories = trait.categories
            if isinstance(categories, basestring):
                categories = [ categories ]
            if type( categories ) in SequenceTypes:
                db_categories = db[ '@categories' ]
                for category in categories:
                    db_categories.setdefault( category, [] )
                    db_categories[ category ].append( name )

        # Close the underlying 'shelve' db (if no batch update is in progress):
        if not self._batch_update:
            self.wdb = None

    #---------------------------------------------------------------------------
    #  Defines a trait as part of a (possibly implicit) package in a database
    #  located in the package's directory:
    #---------------------------------------------------------------------------

    def define ( self, name, trait ):
        """ Defines a trait as part of a (possibly implicit) package in a
            database located in the package's directory.
        """
        # If the name includes an explicit package name, use the specified
        # package name:
        package = None
        col     = name.rfind( '.' )
        if col >= 0:
            package = name[:col]
            name    = name[col+1:]

        # Get the directory name of the caller's module:
        dir = split( inspect.stack(1)[1][1] )[0]
        if dir == '':
            dir = os.getcwd()

        dir = self._normalize( dir )

        if package is None:
            # If no explicit package name specified, see if we have already
            # figured this out before:
            package = self._package_map.get( dir )

            # If not, search the Python path for a valid package:
            if package is None:
                package = self._find_package( dir )

                # If we couldn't find a package, give up:
                if package is None:
                    raise ValueError, ("The 'define' method call should only "
                        "be made from within a module that is part of a "
                        "package, unless an explicit package is specified as "
                        "part of the trait name.")

                # Otherwise, cache the package for the next 'define' call:
                self._package_map[ dir ] = package

        # Keep the trait data base open after the update:
        self._batch_update = True

        # Add the definition to the explicit traits data base contained in the
        # caller's directory:
        file_name      = self.file_name
        self.file_name = join( dir, DB_NAME )
        self( '%s.%s' % ( package, name ), trait )
        self.file_name = file_name

        # Make sure we are registered to do a master data base 'update' on exit:
        if not self._at_exit:
            self._at_exit = True
            atexit.register( self.update )

    #---------------------------------------------------------------------------
    #  Exports all of the traits in a specified package from the master traits
    #  data base to a traits data base in the package's directory, or to a
    #  data base called "'package'_traits_db" in the master traits data base's
    #  directory if the package directory is not writable. The name of the
    #  export file is returned as the result:
    #---------------------------------------------------------------------------

    def export ( self, package ):
        """Exports all of the traits in a specified package from the master
           traits data base to a traits data base in the package's directory, or
           to a data base called <package>_traits_db in the master traits data
           base's directory if the package directory is not writable. The name
           of the export file is returned as the result.
       """
        # Substitute the global package for an empty package name:
        if package == '':
            package = 'global'

        # Get the set of traits to be exported:
        exported = {}
        rdb      = self.rdb
        for name in self.names( package ):
            exported[ name ] = rdb[ name ]
        self.rdb = None

        wdb       = None
        file_name = self.file_name
        if package != 'global':
            # Iterate over all elements of the Python path looking for the
            # matching package directory to export to:
            result = None
            dirs   = package.split( '.' )
            for path in sys.path:
                for dir in dirs:
                    path = join( path, dir )
                    if ((not exists( path )) or
                        (not exists( join( path, '__init__.py' ) ))):
                        break
                else:
                    result = path
                    break

            # If we found the package directory, attempt to set up a writable
            # data base:
            if result is not None:
                self.file_name = result
                try:
                    wdb = self.wdb
                except:
                    pass

        # If we could not create the data base in a package directory, then
        # create it in the master trait data base directory:
        if wdb is None:
            result = join( traits_home(),
                           package.replace( '.', '_' ) + DB_NAME )
            self.file_name = result

        # Copy all of the trait definitions into the export data base:
        self._batch_update = True
        for name, trait in exported.items():
            self( name, trait )

        # Restore the original state and close the export data base:
        self._batch_update = False
        self.wdb           = None
        self.file_name     = file_name

        # Return the name of the export data base as the result:
        return result

    #---------------------------------------------------------------------------
    #  Updates the master data base with the contents of any traits data bases
    #  found in the PythonPath:
    #---------------------------------------------------------------------------

    def update ( self ):
        """ Updates the master database with the contents of any Traits
        databases found in the Python path.
        """
        # Make sure that there is no currently open trait data base:
        self.wdb = None

        # Indicate that the data base should be left open after each
        # transaction:
        self._batch_update = True

        # Get the set of all sub trait data bases contained in the master data
        # base:
        dbs = self.wdb[ '@dbs' ]

        # Iterate over all elements of the Python path looking for packages
        # that contain traits data bases:
        for path in sys.path:
            for root, dirs, files in os.walk( path ):
                if root != path:
                    try:
                        files.index( '__init__.py' )
                        for file in files:
                            if splitext( file )[0] == DB_NAME:
                                time_stamp = os.stat(
                                                join( root, file ) ).st_mtime
                                if dbs.get( root ) != time_stamp:
                                    self._update( root )
                                    dbs[ root ] = time_stamp
                                break
                    except:
                        del dirs[:]

        # Indicate that the data base should no longer be left open after each
        # transaction:
        self._batch_update = False

        # Make sure that the trait data base is closed:
        self.wdb = None

    #---------------------------------------------------------------------------
    #  Returns some or all traits names defined in the data base:
    #---------------------------------------------------------------------------

    def names ( self, package = None ):
        """ Returns some or all trait names defined in the data base.
        """
        names = self.rdb[ '@names' ]
        if package is None:
            return names[:]

        if package[-1:] != '.':
            package += '.'

        n       = len( package )
        result  = []
        matched = False
        for name in names:
            if name[:n] == package:
                matched = True
                if name.rfind( '.' ) < n:
                    result.append( name[n:] )
            elif matched:
                break
        return result

    #---------------------------------------------------------------------------
    #  Returns all the immediate sub-packages of a specified package name:
    #---------------------------------------------------------------------------

    def packages ( self, package = '' ):
        """ Returns all the immediate sub-packages of a specified package name.
        """
        if (len( package ) > 0) and (package[-1:] != '.'):
            package += '.'

        n       = len( package )
        last    = ''
        result  = []
        matched = False
        for name in self.rdb[ '@names' ]:
            if name[:n] == package:
                matched = True
                package = name[n:]
                col     = package.find( '.' )
                if col >= 0:
                    package = package[:col]
                    if package != last:
                        last = package
                        result.append( package )
            elif matched:
                break
        return result

    #---------------------------------------------------------------------------
    #  Returns the names of the traits associated with a specifed category:
    #---------------------------------------------------------------------------

    def categories ( self, category = None ):
        """ Returns the names of the traits associated with a specifed category.
        """
        categories = self.rdb[ '@categories' ]
        if category is None:
            names = categories.keys()
            names.sort()
            return names

        return categories.get( category, [] )

    #---------------------------------------------------------------------------
    #  Returns the names of all traits derived from a specified trait name:
    #---------------------------------------------------------------------------

    def children ( self, parent ):
        """ Returns the names of all traits derived from a specified trait name.
        """
        return self.rdb[ '@children' ].get( self._package_name( parent ), [] )

    #---------------------------------------------------------------------------
    #  Returns the fully qualified package.name form of a specified trait name:
    #---------------------------------------------------------------------------

    def _package_name ( self, name ):
        """ Returns the fully qualified package.name form of a specified trait
            name.
        """
        if name.find( '.' ) >= 0:
            return name
        return '%s.%s' % ( self.default, name )

    #---------------------------------------------------------------------------
    #  Gets the current trait data base file name:
    #---------------------------------------------------------------------------

    def _file_name ( self ):
        """ Gets the current trait data base file name.
        """
        if self.file_name != '':
            return self.file_name
        return join( traits_home(), DB_NAME )

    #---------------------------------------------------------------------------
    #  Tries to find a package that contains the caller's source file:
    #---------------------------------------------------------------------------

    def _find_package ( self, dir ):
        """ Tries to find a package that contains the caller's source file.
        """
        # Search all the directories in the Python path:
        for path in sys.path:
            path = self._normalize( path )
            n    = len( path )
            if ((len( dir ) > n)  and
                (dir[:n] == path) and
                (dir[n:n+1] in '/\\')):

                # Match found, make sure it is really a Python package:
                pdir = dir
                while True:
                    if not exists( join( pdir, '__init__.py' ) ):
                        break
                    pdir = split( pdir )[0]
                    if len( pdir ) <= n:
                        # It really is a package, return the package name:
                        return dir[n+1:].replace('/', '.').replace('\\', '.')

        # No package found:
        return None

    #---------------------------------------------------------------------------
    #  Updates the contents of the master trait data base with the contents of
    #  a specified trait data base (specified by its path):
    #---------------------------------------------------------------------------

    def _update ( self, path ):
        """ Updates the contents of the master trait data base with the contents
            of a specified trait data base (specified by its path).
        """
        try:
            udb = shelve.open( join( path, DB_NAME ), flag     = 'r',
                                                      protocol = -1 )
            for name in udb[ '@names' ]:
                self( name, udb[ name ] )
            udb.close()
        except:
            pass

    #---------------------------------------------------------------------------
    #  Returns a normalized form of a file name:
    #---------------------------------------------------------------------------

    def _normalize ( self, file_name ):
        """ Returns a normalized form of a file name.
        """

        if sys.platform == 'win32':
            return file_name.lower()
        return file_name

#-------------------------------------------------------------------------------
#  Create the singleton Trait data base object:
#-------------------------------------------------------------------------------

tdb = TraitDB()

#-------------------------------------------------------------------------------
#  Handle the user request if we are invoked directly from the command line:
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    if (len( sys.argv ) == 2) and (sys.argv[1] == 'update'):
        tdb.update()
        print "The master traits data base has been updated."
    elif (len( sys.argv ) == 3) and (sys.argv[1] == 'export'):
        file = tdb.export( sys.argv[2] )
        print "Exported package '%s' to: %s." % ( sys.argv[2], file )
    else:
        print "Correct usage is: traits_db.py update"
        print "              or: traits_db.py export package_name"
