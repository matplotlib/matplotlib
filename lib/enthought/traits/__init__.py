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
# Date: 06/21/2002
# Description: Define a 'traits' package that allows other classes to easily
#              define 'type-checked' and/or 'delegated' traits for their
#              instances.
#
#              Note: A 'trait' is similar to a 'property', but is used instead
#              of the word 'property' to differentiate it from the Python
#              language 'property' feature.
#------------------------------------------------------------------------------

try:
    # if the code is ran from an egg, the namespace must be declared
    pass
except:
    pass                                
