"""
==============
Backend Driver
==============

This is used to drive many of the examples across the backends, for
regression testing, and comparing backend efficiency.

You can specify the backends to be tested either via the --backends
switch, which takes a comma-separated list, or as separate arguments,
e.g.

    python backend_driver.py agg ps

would test the agg and ps backends. If no arguments are given, a
default list of backends will be tested.

Interspersed with the backend arguments can be switches for the Python
interpreter executing the tests. If entering such arguments causes an
option parsing error with the driver script, separate them from driver
switches with a --.
"""

import os
import time
import sys
import glob
from optparse import OptionParser

import matplotlib.rcsetup as rcsetup
from matplotlib.cbook import Bunch, dedent


all_backends = list(rcsetup.all_backends)  # to leave the original list alone

# actual physical directory for each dir
dirs = dict(files=os.path.join('..', 'lines_bars_and_markers'),
            shapes=os.path.join('..', 'shapes_and_collections'),
            images=os.path.join('..', 'images_contours_and_fields'),
            pie=os.path.join('..', 'pie_and_polar_charts'),
            text=os.path.join('..', 'text_labels_and_annotations'),
            ticks=os.path.join('..', 'ticks_and_spines'),
            subplots=os.path.join('..', 'subplots_axes_and_figures'),
            specialty=os.path.join('..', 'specialty_plots'),
            showcase=os.path.join('..', 'showcase'),
            pylab=os.path.join('..', 'pylab_examples'),
            api=os.path.join('..', 'api'),
            units=os.path.join('..', 'units'),
            mplot3d=os.path.join('..', 'mplot3d'),
            colors=os.path.join('..', 'color'))


# files in each dir
files = dict()

files['lines'] = [
    'barh.py',
    'cohere.py',
    'fill.py',
    'fill_demo_features.py',
    'line_demo_dash_control.py',
    'line_styles_reference.py',
    'scatter_with_legend.py'
    ]

files['shapes'] = [
    'path_patch_demo.py',
    'scatter_demo.py',
    ]

files['colors'] = [
    'color_cycle_default.py',
    'color_cycle_demo.py',
    ]

files['images'] = [
    'image_demo.py',
    'contourf_log.py',
    ]

files['statistics'] = [
    'errorbar.py',
    'errorbar_features.py',
    'histogram_cumulative.py',
    'histogram_features.py',
    'histogram_histtypes.py',
    'histogram_multihist.py',
    ]

files['pie'] = [
    'pie_demo.py',
    'polar_bar.py',
    'polar_scatter.py',
    ]

files['text_labels_and_annotations'] = [
    'accented_text.py',
    'text_demo_fontdict.py',
    'text_rotation.py',
    'unicode_demo.py',
    ]

files['ticks_and_spines'] = [
    'spines_demo_bounds.py',
    'ticklabels_demo_rotation.py',
    ]

files['subplots_axes_and_figures'] = [
    'subplot_demo.py',
    ]

files['showcase'] = [
    'integral_demo.py',
    ]

files['pylab'] = [
    'alignment_test.py',
    'annotation_demo.py',
    'annotation_demo.py',
    'annotation_demo2.py',
    'annotation_demo2.py',
    'anscombe.py',
    'arctest.py',
    'arrow_demo.py',
    'axes_demo.py',
    'axes_props.py',
    'axhspan_demo.py',
    'axis_equal_demo.py',
    'bar_stacked.py',
    'barb_demo.py',
    'barchart_demo.py',
    'barcode_demo.py',
    'boxplot_demo.py',
    'broken_barh.py',
    'color_by_yvalue.py',
    'color_demo.py',
    'colorbar_tick_labelling_demo.py',
    'contour_demo.py',
    'contour_image.py',
    'contour_label_demo.py',
    'contourf_demo.py',
    'coords_demo.py',
    'coords_report.py',
    'csd_demo.py',
    'cursor_demo.py',
    'custom_cmap.py',
    'custom_figure_class.py',
    'custom_ticker1.py',
    'customize_rc.py',
    'dashpointlabel.py',
    'date_demo_convert.py',
    'date_demo_rrule.py',
    'date_index_formatter.py',
    'dolphin.py',
    'ellipse_collection.py',
    'ellipse_demo.py',
    'ellipse_rotated.py',
    'errorbar_limits.py',
    'fancyarrow_demo.py',
    'fancybox_demo.py',
    'fancybox_demo2.py',
    'fancytextbox_demo.py',
    'figimage_demo.py',
    'figlegend_demo.py',
    'figure_title.py',
    'fill_between_demo.py',
    'fill_spiral.py',
    'findobj_demo.py',
    'fonts_demo.py',
    'fonts_demo_kw.py',
    'ganged_plots.py',
    'geo_demo.py',
    'gradient_bar.py',
    'griddata_demo.py',
    'hatch_demo.py',
    'hexbin_demo.py',
    'hexbin_demo2.py',
    'vline_hline_demo.py',

    'image_clip_path.py',
    'image_demo.py',
    'image_demo2.py',
    'image_interp.py',
    'image_masked.py',
    'image_nonuniform.py',
    'image_origin.py',
    'image_slices_viewer.py',
    'interp_demo.py',
    'invert_axes.py',
    'layer_images.py',
    'legend_demo2.py',
    'legend_demo3.py',
    'line_collection.py',
    'line_collection2.py',
    'log_bar.py',
    'log_demo.py',
    'log_test.py',
    'major_minor_demo1.py',
    'major_minor_demo2.py',
    'masked_demo.py',
    'mathtext_demo.py',
    'mathtext_examples.py',
    'matshow.py',
    'mri_demo.py',
    'mri_with_eeg.py',
    'multi_image.py',
    'multiline.py',
    'multiple_figs_demo.py',
    'nan_test.py',
    'scalarformatter.py',
    'pcolor_demo.py',
    'pcolor_log.py',
    'pcolor_small.py',
    'pie_demo2.py',
    'polar_demo.py',
    'polar_legend.py',
    'psd_demo.py',
    'psd_demo2.py',
    'psd_demo3.py',
    'quadmesh_demo.py',
    'quiver_demo.py',
    'scatter_custom_symbol.py',
    'scatter_demo2.py',
    'scatter_masked.py',
    'scatter_profile.py',
    'scatter_star_poly.py',
    #'set_and_get.py',
    'shared_axis_across_figures.py',
    'shared_axis_demo.py',
    'simple_plot.py',
    'specgram_demo.py',
    'spine_placement_demo.py',
    'spy_demos.py',
    'stem_plot.py',
    'step_demo.py',
    'stix_fonts_demo.py',
    'subplots_adjust.py',
    'symlog_demo.py',
    'table_demo.py',
    'text_rotation_relative_to_line.py',
    'transoffset.py',
    'xcorr_demo.py',
    'zorder_demo.py',
    ]


files['api'] = [
    'agg_oo.py',
    'barchart_demo.py',
    'bbox_intersect.py',
    'collections_demo.py',
    'colorbar_only.py',
    'custom_projection_example.py',
    'custom_scale_example.py',
    'date_demo.py',
    'date_index_formatter.py',
    'donut_demo.py',
    'font_family_rc.py',
    'image_zcoord.py',
    'joinstyle.py',
    'legend_demo.py',
    'line_with_text.py',
    'logo2.py',
    'mathtext_asarray.py',
    'patch_collection.py',
    'quad_bezier.py',
    'scatter_piecharts.py',
    'span_regions.py',
    'two_scales.py',
    'unicode_minus.py',
    'watermark_image.py',
    'watermark_text.py',
]

files['units'] = [
    'annotate_with_units.py',
    #'artist_tests.py',  # broken, fixme
    'bar_demo2.py',
    #'bar_unit_demo.py', # broken, fixme
    #'ellipse_with_units.py',  # broken, fixme
    'radian_demo.py',
    'units_sample.py',
    #'units_scatter.py', # broken, fixme

    ]

files['mplot3d'] = [
    '2dcollections3d_demo.py',
    'bars3d_demo.py',
    'contour3d_demo.py',
    'contour3d_demo2.py',
    'contourf3d_demo.py',
    'lines3d_demo.py',
    'polys3d_demo.py',
    'scatter3d_demo.py',
    'surface3d_demo.py',
    'surface3d_demo2.py',
    'text3d_demo.py',
    'wire3d_demo.py',
    ]

# dict from dir to files we know we don't want to test (e.g., examples
# not using pyplot, examples requiring user input, animation examples,
# examples that may only work in certain environs (usetex examples?),
# examples that generate multiple figures

excluded = {
    'units': ['__init__.py', 'date_support.py', ],
}


def report_missing(dir, flist):
    """Report the .py files in *dir* that are not in *flist*."""
    globstr = os.path.join(dir, '*.py')
    fnames = glob.glob(globstr)

    pyfiles = {os.path.split(fullpath)[-1] for fullpath in fnames}

    exclude = set(excluded.get(dir, []))
    flist = set(flist)
    missing = list(pyfiles - flist - exclude)
    if missing:
        print('%s files not tested: %s' % (dir, ', '.join(sorted(missing))))


def report_all_missing(directories):
    for f in directories:
        report_missing(dirs[f], files[f])


# tests known to fail on a given backend

failbackend = dict(
    svg=('tex_demo.py', ),
    agg=('hyperlinks.py', ),
    pdf=('hyperlinks.py', ),
    ps=('hyperlinks.py', ),
    )


import subprocess


def run(arglist):
    try:
        ret = subprocess.call(arglist)
    except KeyboardInterrupt:
        sys.exit()
    else:
        return ret


def drive(backend, directories, python=['python'], switches=[]):
    exclude = failbackend.get(backend, [])

    # Clear the destination directory for the examples
    path = backend
    if os.path.exists(path):
        for fname in os.listdir(path):
            os.unlink(os.path.join(path, fname))
    else:
        os.mkdir(backend)
    failures = []

    testcases = [os.path.join(dirs[d], fname)
                 for d in directories
                 for fname in files[d]]

    for fullpath in testcases:
        print('\tdriving %-40s' % (fullpath))
        sys.stdout.flush()
        fpath, fname = os.path.split(fullpath)

        if fname in exclude:
            print('\tSkipping %s, known to fail on backend: %s' % backend)
            continue

        basename, ext = os.path.splitext(fname)
        outfile = os.path.join(path, basename)
        tmpfile_name = '_tmp_%s.py' % basename
        tmpfile = open(tmpfile_name, 'w')

        for line in open(fullpath):
            line_lstrip = line.lstrip()
            if line_lstrip.startswith("#"):
                tmpfile.write(line)

        tmpfile.writelines((
            'import sys\n',
            'sys.path.append("%s")\n' % fpath.replace('\\', '\\\\'),
            'import matplotlib\n',
            'matplotlib.use("%s")\n' % backend,
            'from pylab import savefig\n',
            'import numpy\n',
            'numpy.seterr(invalid="ignore")\n',
            ))
        for line in open(fullpath):
            if line.lstrip().startswith(('matplotlib.use', 'savefig', 'show')):
                continue
            tmpfile.write(line)
        if backend in rcsetup.interactive_bk:
            tmpfile.write('show()')
        else:
            tmpfile.write('\nsavefig(r"%s", dpi=150)' % outfile)

        tmpfile.close()
        start_time = time.time()
        program = [x % {'name': basename} for x in python]
        ret = run(program + [tmpfile_name] + switches)
        end_time = time.time()
        print("%s %s" % ((end_time - start_time), ret))
        # subprocess.call([python, tmpfile_name] + switches)
        os.remove(tmpfile_name)
        if ret:
            failures.append(fullpath)
    return failures


def parse_options():
    doc = __doc__.split("\n\n") if __doc__ else "  "
    op = OptionParser(description=doc[0].strip(),
                      usage='%prog [options] [--] [backends and switches]',
                      epilog='\n'.join(doc[1:]))
    op.disable_interspersed_args()
    op.set_defaults(dirs='pylab,api,units,mplot3d',
                    clean=False, coverage=False, valgrind=False)
    op.add_option('-d', '--dirs', '--directories', type='string',
                  dest='dirs', help=dedent('''
      Run only the tests in these directories; comma-separated list of
      one or more of: pylab (or pylab_examples), api, units, mplot3d'''))
    op.add_option('-b', '--backends', type='string', dest='backends',
                  help=dedent('''
      Run tests only for these backends; comma-separated list of
      one or more of: agg, ps, svg, pdf, template, cairo,
      Default is everything except cairo.'''))
    op.add_option('--clean', action='store_true', dest='clean',
                  help='Remove result directories, run no tests')
    op.add_option('-c', '--coverage', action='store_true', dest='coverage',
                  help='Run in coverage.py')
    op.add_option('-v', '--valgrind', action='store_true', dest='valgrind',
                  help='Run in valgrind')

    options, args = op.parse_args()
    switches = [x for x in args if x.startswith('--')]
    backends = [x.lower() for x in args if not x.startswith('--')]
    if options.backends:
        backends += [be.lower() for be in options.backends.split(',')]

    result = Bunch(
        dirs=options.dirs.split(','),
        backends=backends or ['agg', 'ps', 'svg', 'pdf', 'template'],
        clean=options.clean,
        coverage=options.coverage,
        valgrind=options.valgrind,
        switches=switches)
    if 'pylab_examples' in result.dirs:
        result.dirs[result.dirs.index('pylab_examples')] = 'pylab'
    return result


if __name__ == '__main__':
    times = {}
    failures = {}
    options = parse_options()

    if options.clean:
        localdirs = [d for d in glob.glob('*') if os.path.isdir(d)]
        all_backends_set = set(all_backends)
        for d in localdirs:
            if d.lower() not in all_backends_set:
                continue
            print('removing %s' % d)
            for fname in glob.glob(os.path.join(d, '*')):
                os.remove(fname)
            os.rmdir(d)
        for fname in glob.glob('_tmp*.py'):
            os.remove(fname)

        print('all clean...')
        raise SystemExit
    if options.coverage:
        python = ['coverage.py', '-x']
    elif options.valgrind:
        python = ['valgrind', '--tool=memcheck', '--leak-check=yes',
                  '--log-file=%(name)s', sys.executable]
    elif sys.platform == 'win32':
        python = [sys.executable]
    else:
        python = [sys.executable]

    report_all_missing(options.dirs)
    for backend in options.backends:
        print('testing %s %s' % (backend, ' '.join(options.switches)))
        t0 = time.time()
        failures[backend] = \
            drive(backend, options.dirs, python, options.switches)
        t1 = time.time()
        times[backend] = (t1 - t0) / 60

    for backend, elapsed in times.items():
        print('Backend %s took %1.2f minutes to complete' % (backend, elapsed))
        failed = failures[backend]
        if failed:
            print('  Failures: %s' % failed)
        if 'template' in times:
            print('\ttemplate ratio %1.3f, template residual %1.3f' % (
                elapsed/times['template'], elapsed - times['template']))
