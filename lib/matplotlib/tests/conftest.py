
from matplotlib.testing.conftest import (  # noqa
    mpl_test_settings, pytest_configure, pytest_unconfigure, pd, xr)

#Global boolean list declarations
histBranchBools = [False for i in range(72)]
#another list with false booleans

tableBranchBools = [False for i in range(28)]


def writeRes():
    currTakenBranches = 0
    global histBranchBools
    f = open("BranchCovRes.txt", "w")
    for currBranchNr in range(len(histBranchBools)):
        if(currBranchNr % 4 == 0 and currBranchNr != 0):
            f.write("||\n")
        f.write(f"|| Branch {currBranchNr} taken: {histBranchBools[currBranchNr]} ")
        if(histBranchBools[currBranchNr]):
            currTakenBranches += 1
    
    f.write("||\n")
    f.write(f"The hist() function took {100*(currTakenBranches/len(histBranchBools))}% of its branches, {currTakenBranches} out of {len(histBranchBools)}, during the tests.")
    f.write("\n")

    currTakenBranches = 0
    global tableBranchBools
    for currBranchNr in range(len(tableBranchBools)):
        if (currBranchNr % 4 == 0 and currBranchNr != 0):
            f.write("||\n")
        f.write(f"|| Branch {currBranchNr} taken: {tableBranchBools[currBranchNr]} ")
        if (tableBranchBools[currBranchNr]):
            currTakenBranches += 1

    f.write("||\n")
    f.write(
        f"The table() function took {100 * (currTakenBranches / len(tableBranchBools))}% of its branches, {currTakenBranches} out of {len(tableBranchBools)}, during the tests.")
    f.write("\n")
    f.close()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    writeRes()