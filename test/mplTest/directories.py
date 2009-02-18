#=======================================================================
"""Default directories for the matplotlib unit-test structure."""
#=======================================================================

import os.path

#=======================================================================
saveDirName = "saved-results"
inputDirName = "inputs"
outputDirName = "outputs"
baselineDirName = "baseline"

#-----------------------------------------------------------------------
def baselineFile( fname ):
   return os.path.join( baselineDirName, fname )

