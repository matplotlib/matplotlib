#!/usr/bin/env python
"""
This is used to drive many of the examples across the backends, for
regression testing, and comparing backend efficiency
"""

from __future__ import division
import os, time, sys
files = (
    'alignment_test.py',
    'arctest.py',
    'axes_demo.py',
    'bar_stacked.py',
    'barchart_demo.py',
    'color_demo.py',
    'contour_demo.py',
    'contourf_demo.py',
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
    'line_collection.py',
    'line_collection2.py',
    'line_styles.py',
    'log_demo.py',
    'log_test.py',
    'major_minor_demo1.py',
    'major_minor_demo2.py',
    'masked_demo.py',
    'mathtext_demo.py',
    'mri_with_eeg.py',
    'multiple_figs_demo.py',
    'pcolor_demo.py',
    'pcolor_demo2.py',
    'pcolor_small.py',
    'pie_demo.py',
    'polar_demo.py',
    'polar_scatter.py',
    'psd_demo.py',
    'quiver_demo.py',
    'scatter_demo.py',
    'scatter_demo2.py',
    'simple_plot.py',
    'specgram_demo.py',
    'stock_demo.py',
    'subplot_demo.py',
#    'set_and_get.py',
    'table_demo.py',
    'text_handles.py',
    'text_rotation.py',
    'text_themes.py',
#    'tex_demo.py',
    'two_scales.py',
    'unicode_demo.py',
    'vline_demo.py',
    'xcorr_demo.py',
    'zorder_demo.py',
    )


#tests known to fail on python22 (require datetime)
fail22  = (
    'date_demo1.py',
    'date_demo2.py',
    'finance_demo.py',
    )


# tests known to fail on a given backend

failbackend = dict(
    SVG = ('tex_demo.py,'),
    )

def drive(backend, python='python'):

    exclude = failbackend.get(backend, [])

    for fname in files:
        if fname in exclude:
            print '\tSkipping %s, known to fail on backend: %s'%backend
            continue

        print '\tdriving %s' % fname
        basename, ext = os.path.splitext(fname)
        outfile = basename + '_%s'%backend
        tmpfile_name = '_tmp_%s.py' % basename
        tmpfile = file(tmpfile_name, 'w')

        tmpfile.writelines((
            'from __future__ import division\n',
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % backend,
            'from pylab import savefig\n',
            ))
        for line in file(fname):
            line_lstrip = line.lstrip()
            if (line_lstrip.startswith('from __future__ import division') or
                line_lstrip.startswith('matplotlib.use') or
                line_lstrip.startswith('savefig') or
                line_lstrip.startswith('show')):
                continue
            tmpfile.write(line)
        if backend in ('GTK', 'WX', 'TkAgg'):
            tmpfile.write('show()')
        else:
            tmpfile.write('savefig("%s", dpi=150)' % outfile)

        tmpfile.close()
        os.system('%s %s' % (python, tmpfile_name))
        os.remove(tmpfile_name)


if __name__ == '__main__':
    times = {}
    # backends = ['Agg', 'Cairo', 'GDK', 'PS', 'SVG', 'Template']
    #backends = ['Agg', 'PS', 'SVG', 'Template']
    # backends = [ 'GTK', 'WX', 'TkAgg']
    default_backends = ['Agg', 'PS', 'SVG', 'Template']
    #default_backends = ['Agg']
    #backends = ['Agg']
    if sys.platform == 'win32':
        python = r'c:\Python24\python.exe'
    else:
        python = 'python'
    if sys.argv[1:]:
        backends = [b for b in sys.argv[1:] if b in default_backends]
    else:
        backends = default_backends
    for backend in backends:
        print 'testing %s' % backend
        t0 = time.time()
        drive(backend, python)
        t1 = time.time()
        times[backend] = (t1-t0)/60.0

    # print times
    for backend, elapsed in times.items():
        print 'Backend %s took %1.2f minutes to complete' % ( backend, elapsed)
        if 'Template' in times:
            print '\ttemplate ratio %1.3f, template residual %1.3f' % (
                elapsed/times['Template'], elapsed-times['Template'])
