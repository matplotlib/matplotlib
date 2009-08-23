import os, sys, glob, shutil
import matplotlib.cbook as cbook

savedresults_dir = 'saved-results'
baseline_dir = 'baseline'
diff_dir = 'diff-images'
basename = 'failed-diff-'
nbase = len(basename)

failed = set()
for fname in glob.glob(os.path.join(diff_dir, '%s*.png'%basename)):
    ind = fname.find(basename)
    fname = fname[ind+nbase:]
    failed.add(fname)

datad = dict()
for fpath in cbook.get_recursive_filelist('.'):
    if not fpath.endswith('.png'): continue
    if fpath.find(diff_dir)>0: continue
    rel_dir, fname = os.path.split(fpath)


    saved = fpath.find(savedresults_dir)>0
    baseline = fpath.find(baseline_dir)>0

    if saved:
        datad.setdefault(fname, [None,None])[0] = fpath
    elif baseline:
        datad.setdefault(fname, [None,None])[1] = fpath

nfailed = len(failed)
for ithis, fname in enumerate(sorted(failed)):
    data = datad.get(fname)
    if data is not None:
        saved, baseline = data
        #print ithis, fname, saved, baseline
        if saved is None:
            print 'could not find saved data for', fname
        elif baseline is None:
            print 'could not find baseline data for', fname
        else:
            x = raw_input('Copy %d of %d\n    saved="%s" to\n    baseline="%s" (n|Y):'%(ithis, nfailed, saved, baseline))
            if x.lower()=='y' or x=='':
                shutil.copy(saved, baseline)
                print '    copied'
            elif x.lower()=='n':
                print '    skipping'
            else:
                print '    skipping unrecognized response="%s"'%x
            print

    else:
        print 'could not find data for', fname
