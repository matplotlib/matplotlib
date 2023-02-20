
from matplotlib.testing.conftest import (  # noqa
    mpl_test_settings, pytest_configure, pytest_unconfigure, pd, xr)

#Global boolean list declarations
histBranchBools = [False for i in range(71)]
#another list with false booleans
histBranchBoolsKey = [False for i in range(23)]
streamBranchBools = [False for i in range(29)]

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
    global streamBranchBools
    f = open("BranchCovRes.txt", "w")
    for currBranchNr in range(len(streamBranchBools)):
        if (currBranchNr % 4 == 0 and currBranchNr != 0):
            f.write("||\n")
        f.write(f"|| Branch {currBranchNr} taken: {streamBranchBools[currBranchNr]} ")
        if (streamBranchBools[currBranchNr]):
            currTakenBranches += 1

    f.write("||\n")
    f.write(
        f"The hist() function took {100 * (currTakenBranches / len(streamBranchBools))}% of its branches, {currTakenBranches} out of {len(streamBranchBools)}, during the tests.")
    f.write("\n")


    f.close()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    writeRes()