"""For all failed image comparisons, gather the baseline image, the
current image and the absdiff image into a single directory specified
by target_dir.

This is meant to be run from the mplroot directory."""

import os, shutil

roots = ['test_matplotlib','test_plots']
savedresults_dir = 'saved-results'
baseline_dir = 'baseline'
expected_basename = 'expected-'
diff_basename = 'failed-diff-'
target_dir = os.path.abspath('status_images')
nbase = len(diff_basename)

def listFiles(root, patterns='*', recurse=1, return_folders=0):
    """
    Recursively list files

    from Parmar and Martelli in the Python Cookbook
    """
    import os.path, fnmatch
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    # Collect input and output arguments into one bunch
    class Bunch:
        def __init__(self, **kwds): self.__dict__.update(kwds)
    arg = Bunch(recurse=recurse, pattern_list=pattern_list,
        return_folders=return_folders, results=[])

    def visit(arg, dirname, files):
        # Append to arg.results all relevant files (and perhaps folders)
        for name in files:
            fullname = os.path.normpath(os.path.join(dirname, name))
            if arg.return_folders or os.path.isfile(fullname):
                for pattern in arg.pattern_list:
                    if fnmatch.fnmatch(name, pattern):
                        arg.results.append(fullname)
                        break
        # Block recursion if recursion was disallowed
        if not arg.recurse: files[:]=[]

    os.path.walk(root, visit, arg)

    return arg.results

def get_recursive_filelist(args):
    """
    Recurse all the files and dirs in *args* ignoring symbolic links
    and return the files as a list of strings
    """
    files = []

    for arg in args:
        if os.path.isfile(arg):
            files.append(arg)
            continue
        if os.path.isdir(arg):
            newfiles = listFiles(arg, recurse=1, return_folders=1)
            files.extend(newfiles)

    return [f for f in files if not os.path.islink(f)]

def path_split_all(fname):
    """split a file path into a list of directories and filename"""
    pieces = [fname]
    previous_tails = []
    while 1:
        head,tail = os.path.split(pieces[0])
        if head=='':
            return pieces + previous_tails
        pieces = [head]
        previous_tails.insert(0,tail)

if 1:
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs( target_dir ) # prevent buildbot DirectoryUpload failure

    # new matplotlib.testing infrastructure

    for fname in get_recursive_filelist(['result_images']):
        # only images
        if not fname.endswith('.png'): continue

        result_dir, result_fname = os.path.split(fname)
        absdiff_fname = os.path.join( result_dir, diff_basename + result_fname)
        expected_fname = os.path.join( result_dir, expected_basename + result_fname)
        if not os.path.exists(absdiff_fname):
            continue
        if not os.path.exists(expected_fname):
            continue
        print fname
        print absdiff_fname

        teststr = os.path.splitext(fname)[0]
        this_targetdir = os.path.join(target_dir,teststr)
        os.makedirs( this_targetdir )
        shutil.copy( expected_fname,
                     os.path.join(this_targetdir,'baseline.png') )
        shutil.copy( fname,
                     os.path.join(this_targetdir,'actual.png') )
        shutil.copy( absdiff_fname,
                     os.path.join(this_targetdir,'absdiff.png') )

    # old mplTest infrastructure
    for fpath in get_recursive_filelist(roots):
        # only images
        if not fpath.endswith('.png'): continue

        pieces = path_split_all( fpath )
        if pieces[1]!=savedresults_dir:
            continue
        root = pieces[0]
        testclass = pieces[2]
        fname = pieces[3]
        if not fname.startswith(diff_basename):
            # only failed images
            continue
        origname = fname[nbase:]
        testname = os.path.splitext(origname)[0]

        # make a name for the test
        teststr = '%s.%s.%s'%(root,testclass,testname)
        this_targetdir = os.path.join(target_dir,teststr)
        os.makedirs( this_targetdir )
        shutil.copy( os.path.join(root,baseline_dir,testclass,origname),
                     os.path.join(this_targetdir,'baseline.png') )
        shutil.copy( os.path.join(root,savedresults_dir,testclass,origname),
                     os.path.join(this_targetdir,'actual.png') )
        shutil.copy( os.path.join(root,savedresults_dir,testclass,fname),
                     os.path.join(this_targetdir,'absdiff.png') )
