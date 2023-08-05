#!/usr/bin/env python

import gc
from io import BytesIO
import tracemalloc

try:
    import psutil
except ImportError as err:
    raise ImportError("This script requires psutil") from err

import numpy as np


def run_memleak_test(bench, iterations, report):
    tracemalloc.start()

    starti = min(50, iterations // 2)
    endi = iterations

    malloc_arr = np.empty(endi, dtype=np.int64)
    rss_arr = np.empty(endi, dtype=np.int64)
    rss_peaks = np.empty(endi, dtype=np.int64)
    nobjs_arr = np.empty(endi, dtype=np.int64)
    garbage_arr = np.empty(endi, dtype=np.int64)
    open_files_arr = np.empty(endi, dtype=np.int64)
    rss_peak = 0

    p = psutil.Process()

    for i in range(endi):
        bench()

        gc.collect()

        rss = p.memory_info().rss
        malloc, peak = tracemalloc.get_traced_memory()
        nobjs = len(gc.get_objects())
        garbage = len(gc.garbage)
        open_files = len(p.open_files())
        print(f"{i: 4d}: pymalloc {malloc: 10d}, rss {rss: 10d}, "
              f"nobjs {nobjs: 10d}, garbage {garbage: 4d}, "
              f"files: {open_files: 4d}")
        if i == starti:
            print(f'{" warmup done ":-^86s}')
        malloc_arr[i] = malloc
        rss_arr[i] = rss
        if rss > rss_peak:
            rss_peak = rss
        rss_peaks[i] = rss_peak
        nobjs_arr[i] = nobjs
        garbage_arr[i] = garbage
        open_files_arr[i] = open_files

    print('Average memory consumed per loop: {:1.4f} bytes\n'.format(
        np.sum(rss_peaks[starti+1:] - rss_peaks[starti:-1]) / (endi - starti)))

    from matplotlib import pyplot as plt
    from matplotlib.ticker import EngFormatter
    bytes_formatter = EngFormatter(unit='B')
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    for ax in (ax1, ax2, ax3):
        ax.axvline(starti, linestyle='--', color='k')
    ax1b = ax1.twinx()
    ax1b.yaxis.set_major_formatter(bytes_formatter)
    ax1.plot(malloc_arr, 'C0')
    ax1b.plot(rss_arr, 'C1', label='rss')
    ax1b.plot(rss_peaks, 'C1', linestyle='--', label='rss max')
    ax1.set_ylabel('pymalloc', color='C0')
    ax1b.set_ylabel('rss', color='C1')
    ax1b.legend()

    ax2b = ax2.twinx()
    ax2.plot(nobjs_arr, 'C0')
    ax2b.plot(garbage_arr, 'C1')
    ax2.set_ylabel('total objects', color='C0')
    ax2b.set_ylabel('garbage objects', color='C1')

    ax3.plot(open_files_arr)
    ax3.set_ylabel('open file handles')

    if not report.endswith('.pdf'):
        report = report + '.pdf'
    fig.tight_layout()
    fig.savefig(report, format='pdf')


class MemleakTest:
    def __init__(self, empty):
        self.empty = empty

    def __call__(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(1)

        if not self.empty:
            t1 = np.arange(0.0, 2.0, 0.01)
            y1 = np.sin(2 * np.pi * t1)
            y2 = np.random.rand(len(t1))
            X = np.random.rand(50, 50)

            ax = fig.add_subplot(221)
            ax.plot(t1, y1, '-')
            ax.plot(t1, y2, 's')

            ax = fig.add_subplot(222)
            ax.imshow(X)

            ax = fig.add_subplot(223)
            ax.scatter(np.random.rand(50), np.random.rand(50),
                       s=100 * np.random.rand(50), c=np.random.rand(50))

            ax = fig.add_subplot(224)
            ax.pcolor(10 * np.random.rand(50, 50))

        fig.savefig(BytesIO(), dpi=75)
        fig.canvas.flush_events()
        plt.close(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Run memory leak tests')
    parser.add_argument('backend', type=str, nargs=1,
                        help='backend to test')
    parser.add_argument('iterations', type=int, nargs=1,
                        help='number of iterations')
    parser.add_argument('report', type=str, nargs=1,
                        help='filename to save report')
    parser.add_argument('--empty', action='store_true',
                        help="Don't plot any content, just test creating "
                        "and destroying figures")
    parser.add_argument('--interactive', action='store_true',
                        help="Turn on interactive mode to actually open "
                        "windows.  Only works with some GUI backends.")

    args = parser.parse_args()

    import matplotlib
    matplotlib.use(args.backend[0])

    if args.interactive:
        import matplotlib.pyplot as plt
        plt.ion()

    run_memleak_test(
        MemleakTest(args.empty), args.iterations[0], args.report[0])
