#!/usr/bin/env python
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
    'custom_ticker1.py',
    'customize_rc.py',
    'date_demo1.py',
    'date_demo2.py',    
    'figimage_demo.py',
    'figlegend_demo.py',
    'figtext.py',
    'fill_demo.py',
    'finance_demo.py',
#    'fonts_demo_kw.py',
    'histogram_demo.py',
    'image_demo.py',
    'image_demo2.py',
    'image_demo_na.py',
    'image_origin.py',
    'invert_axes.py',
    'layer_images.py',
    'legend_demo.py',
    'legend_demo2.py',
    'line_styles.py',
    'log_demo.py',
    'log_test.py',
    'major_minor_demo1.py',
    'major_minor_demo2.py',     
    'mathtext_demo.py',
    'mri_with_eeg.py',
    'multiple_figs_demo.py',
    'pcolor_demo.py',
    'pcolor_demo2.py',
    'pcolor_small.py',
    'polar_demo.py',
    'polar_scatter.py',
    'psd_demo.py',
    'scatter_demo.py',
    'scatter_demo2.py',
    'simple_plot.py',
    'specgram_demo.py',
    'stock_demo.py',
    'subplot_demo.py',
#    'set_and_get.py',    
    'table_demo.py',
    'text_handles.py',
    'text_themes.py',
    'two_scales.py',
    'vline_demo.py',
    )


#tests known to fail on python22 (require datetime)
fail22  = (
    'date_demo1.py',
    'date_demo2.py',    
    'finance_demo.py',
    )
def drive(backend, python='python2.3'):
    
    for fname in files:
        if python=='python2.2' and fname in fail22:
            print '\tSkipping %s, known to fail on python2.2'%fname
            continue
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
        os.system('%s %s' % (python, tmpfile))

times = {}
backends = ['PS', 'GD', 'Paint', 'Agg', 'Template']
#backends.extend([ 'GTK', 'WX', 'TkAgg'])
#backends = [ 'Agg']
backends = ['PS', 'Agg']

python = 'python2.3'
for backend in backends:
    print 'testing %s' % backend
    t0 = time.time()
    drive(backend, python)
    t1 = time.time()
    times[backend] = (t1-t0)/60.0

#print times
for backend, elapsed in times.items():
    print 'Backend %s took %1.2f minutes to complete' % ( backend, elapsed)
    print '\ttemplate ratio %1.3f, template residual %1.3f' % (
        elapsed/times['Template'], elapsed-times['Template'])
    
