"""
This is use to drive many of the examples across the image backends
and is used for regression testing and comparing backend efficiency

This example creates a lot of temp files name _tmp_*.py.  You'll
probably want to remove them after the script runs

"""

from __future__ import division
import os, time
files = (
    'alignment_test.py',
    'arctest.py',
    'axes_demo.py',
    'bar_stacked.py',
    'barchart_demo.py',
    'color_demo.py',
    'csd_demo.py',
    'fill_demo.py',
    'figtext.py',
    'histogram_demo.py',
    'image_demo.py',
    'image_demo2.py',
    'legend_demo.py',
    'legend_demo2.py',
    'line_styles.py',
    'log_demo.py',
    'log_test.py',
    'mathtext_demo.py',
    'mri_with_eeg.py',
    'multiple_figs_demo.py',
    'pcolor_demo.py',
    'pcolor_demo2.py',
    'psd_demo.py',
    'scatter_demo.py',
    'scatter_demo2.py',
    'simple_plot.py',
    'stock_demo.py',
    'specgram_demo.py',
    'subplot_demo.py',
    'table_demo.py',
    'text_handles.py',
    'text_themes.py',
    )

def drive(backend):

    for fname in files:
        lines = [
            'from __future__ import division\n',
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % backend]
        print '\tdriving %s' % fname
        for line in file(fname):
            if line.strip().startswith('from __future__ import division'): continue
            if line.strip().startswith('matplotlib.use'): continue
            if line.strip().startswith('savefig'): continue
            if line.strip().startswith('show'): continue
            lines.append(line)
        basename, ext = os.path.splitext(fname)
        outfile = basename + '_%s'%backend
        if backend in ('GTK', 'WX', 'TkAgg'):
            lines.append('show()')
        else:
            lines.append('savefig("%s", dpi=150)' % outfile)
        tmpfile = '_tmp_%s.py' % basename
        file(tmpfile, 'w').write(''.join(lines))
        os.system('python %s' % tmpfile)

times = {}
backends = ['PS', 'GD', 'Paint', 'Agg', 'Template']
#backends.extend([ 'GTK', 'WX', 'TkAgg'])
#backends = [ 'Agg']

for backend in backends:
    print 'testing %s' % backend
    t0 = time.time()
    drive(backend)
    t1 = time.time()
    times[backend] = (t1-t0)/60.0

#print times
for backend, elapsed in times.items():
    print 'Backend %s took %1.2f minutes to complete' % ( backend, elapsed)
    print '\ttemplate ratio %1.3f, template residual %1.3f' % (
        elapsed/times['Template'], elapsed-times['Template'])
    
