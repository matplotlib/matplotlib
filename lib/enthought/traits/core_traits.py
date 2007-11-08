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
# Date: 11/26/2004
# ------------------------------------------------------------------------------
""" Adds all of the core traits to the Traits database.
"""
if __name__ == '__main__':
    
    from enthought.traits.api import Event, List, Dict, Any, Int, Long, Float, Str
    from enthought.traits.api import Unicode, Complex, Bool, CInt, CLong, CFloat
    from enthought.traits.api import CStr, CUnicode, CComplex, CBool, false, true
    from enthought.traits.api import String, Password, File, Directory, Function
    from enthought.traits.api import Method, Class, Module, Type, This, self, Python
    from enthought.traits.api import ReadOnly, ListInt, ListFloat, ListStr
    from enthought.traits.api import ListUnicode, ListComplex, ListBool
    from enthought.traits.api import ListFunction, ListMethod, ListClass
    from enthought.traits.api import ListInstance, ListThis, DictStrAny, DictStrStr
    from enthought.traits.api import DictStrInt, DictStrLong, DictStrFloat
    from enthought.traits.api import DictStrBool,DictStrList
    from enthought.traits.api import tdb
         
    define = tdb.define
    define( 'Event',        Event )
    define( 'List',         List )
    define( 'Dict',         Dict )
    define( 'Any',          Any )
    define( 'Int',          Int )
    define( 'Long',         Long )
    define( 'Float',        Float )
    define( 'Str',          Str )
    define( 'Unicode',      Unicode )
    define( 'Complex',      Complex )
    define( 'Bool',         Bool )
    define( 'CInt',         CInt )
    define( 'CLong',        CLong )
    define( 'CFloat',       CFloat )
    define( 'CStr',         CStr )
    define( 'CUnicode',     CUnicode )
    define( 'CComplex',     CComplex )
    define( 'CBool',        CBool )
    define( 'false',        false )
    define( 'true',         true )
    define( 'String',       String )
    define( 'Password',     Password )
    define( 'File',         File )
    define( 'Directory',    Directory )
#   define( 'Function',     Function )
#   define( 'Method',       Method )
#   define( 'Class',        Class )
#   define( 'Module',       Module )
    define( 'Type',         Type )
    define( 'This',         This )
#   define( 'self',         self )
    define( 'Python',       Python )
##  define( 'ReadOnly',     ReadOnly ) <-- 'Undefined' doesn't have right
                                         # semantics when persisted
    define( 'ListInt',      ListInt )
    define( 'ListFloat',    ListFloat )
    define( 'ListStr',      ListStr )
    define( 'ListUnicode',  ListUnicode )
    define( 'ListComplex',  ListComplex )
    define( 'ListBool',     ListBool )
#   define( 'ListFunction', ListFunction )
#   define( 'ListMethod',   ListMethod )
#   define( 'ListClass',    ListClass )
#   define( 'ListInstance', ListInstance )
    define( 'ListThis',     ListThis )
    define( 'DictStrAny',   DictStrAny )
    define( 'DictStrStr',   DictStrStr )
    define( 'DictStrInt',   DictStrInt )
    define( 'DictStrLong',  DictStrLong )
    define( 'DictStrFloat', DictStrFloat )
    define( 'DictStrBool',  DictStrBool )
    define( 'DictStrList',  DictStrList )
    
