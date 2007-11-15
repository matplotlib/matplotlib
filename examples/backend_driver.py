#!/usr/bin/env python
"""
This is used to drive many of the examples across the backends, for
regression testing, and comparing backend efficiency.

The script takes one or more arguments specifying backends
to be tested, e.g.

    python backend_driver.py agg ps cairo.png cairo.ps

would test the agg and ps backends, and the cairo backend with
output to png and ps files.

If no arguments are given, a default list of backends will be
tested.
"""

from __future__ import division
import os, time, sys
import matplotlib.backends as mplbe

files = (
    'alignment_test.py',
    'arctest.py',
    'arrow_demo.py',
    'axes_demo.py',
    'axhspan_demo.py',
    'bar_stacked.py',
    'barchart_demo.py',
    'boxplot_demo.py',
    'broken_barh.py',
    'barh_demo.py',
    'color_demo.py',
    'colorbar_only.py',
    'cohere_demo.py',
    'contour_demo.py',
    'contourf_demo.py',
    'csd_demo.py',    
    'custom_ticker1.py',
    'customize_rc.py',
    'date_demo1.py',
    'date_demo2.py',
    'equal_aspect_ratio.py',
    'errorbar_limits.py',
    'figimage_demo.py',
    'figlegend_demo.py',
    'figtext.py',
    'fill_demo.py',
    'finance_demo.py',
    'fonts_demo_kw.py',
    'histogram_demo.py',
    'hline_demo.py',
    'image_demo.py',
    'image_demo2.py',
    'image_masked.py',
    'image_origin.py',
    'invert_axes.py',
    'layer_images.py',
    'legend_auto.py',
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
    'quadmesh_demo.py',
    'quiver_demo.py',
    'scatter_demo.py',
    'scatter_demo2.py',
    'scatter_star_poly.py',
    'shared_axis_demo.py',
    'shared_axis_across_figures.py',
    'simple_plot.py',
    'specgram_demo.py',
    'spy_demos.py',
    'stem_plot.py',
    'step_demo.py',
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


# tests known to fail on a given backend

failbackend = dict(
    SVG = ('tex_demo.py,'),
    )

try:
    import subprocess
    def run(arglist):
        try:
            subprocess.call(arglist)
        except KeyboardInterrupt:
            sys.exit()
except ImportError:
    def run(arglist):
        os.system(' '.join(arglist))

def drive(backend, python=['python'], switches = []):
    exclude = failbackend.get(backend, [])
    # Strip off the format specifier, if any.
    if backend.startswith('cairo'):
        _backend = 'cairo'
    else:
        _backend = backend
    for fname in files:
        if fname in exclude:
            print '\tSkipping %s, known to fail on backend: %s'%backend
            continue

        print ('\tdriving %-40s' % (fname)),
        basename, ext = os.path.splitext(fname)
        outfile = basename + '_%s'%backend
        tmpfile_name = '_tmp_%s.py' % basename
        tmpfile = file(tmpfile_name, 'w')

        for line in file(fname):
            line_lstrip = line.lstrip()
            if line_lstrip.startswith("#"):
                tmpfile.write(line)
            else:
                break

        tmpfile.writelines((
            'from __future__ import division\n',
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % _backend,
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
        if backend in mplbe.interactive_bk:
            tmpfile.write('show()')
        else:
            tmpfile.write('savefig("%s", dpi=150)' % outfile)

        tmpfile.close()
        start_time = time.time()
        run(python + [tmpfile_name, switchstring])
        end_time = time.time()
        print (end_time - start_time)
        #os.system('%s %s %s' % (python, tmpfile_name, switchstring))
        os.remove(tmpfile_name)

if __name__ == '__main__':
    times = {}
    default_backends = ['Agg', 'PS', 'SVG', 'PDF', 'Template']
    if '--coverage' in sys.argv:
        python = ['coverage.py', '-x']
        sys.argv.remove('--coverage')
    elif sys.platform == 'win32':
        python = [r'c:\Python24\python.exe']
    else:
        python = ['python']
    all_backends = [b.lower() for b in mplbe.all_backends]
    all_backends.extend(['cairo.png', 'cairo.ps', 'cairo.pdf', 'cairo.svg'])
    backends = []
    switches = []
    if sys.argv[1:]:
        backends = [b.lower() for b in sys.argv[1:] if b.lower() in all_backends]
        switches = [s for s in sys.argv[1:] if s.startswith('--')]
    if not backends:
        backends = default_backends
    for backend in backends:
        switchstring = ' '.join(switches)
        print 'testing %s %s' % (backend, switchstring)
        t0 = time.time()
        drive(backend, python, switches)
        t1 = time.time()
        times[backend] = (t1-t0)/60.0

    # print times
    for backend, elapsed in times.items():
        print 'Backend %s took %1.2f minutes to complete' % ( backend, elapsed)
        if 'Template' in times:
            print '\ttemplate ratio %1.3f, template residual %1.3f' % (
                elapsed/times['Template'], elapsed-times['Template'])
