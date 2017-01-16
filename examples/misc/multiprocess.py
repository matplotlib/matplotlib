# -*- noplot -*-
# Demo of using multiprocessing for generating data in one process and plotting
# in another.
# Written by Robert Cimrman

from __future__ import print_function
from six.moves import input

import time
from multiprocessing import Process, Pipe
import numpy as np

import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import gobject

# Fixing random state for reproducibility
np.random.seed(19680801)


class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def poll_draw(self):

        def call_back():
            while self.pipe.poll():
                command = self.pipe.recv()
                if command is None:
                    self.terminate()
                    return False
                else:
                    self.x.append(command[0])
                    self.y.append(command[1])
                    self.ax.plot(self.x, self.y, 'ro')
            self.fig.canvas.draw()
            return True

        return call_back

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        self.gid = gobject.timeout_add(1000, self.poll_draw())

        print('...done')
        plt.show()


class NBPlot(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = Process(target=self.plotter,
                                    args=(plotter_pipe,))
        self.plot_process.daemon = True
        self.plot_process.start()

    def plot(self, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            data = np.random.random(2)
            send(data)


def main():
    pl = NBPlot()
    for ii in range(10):
        pl.plot()
        time.sleep(0.5)
    input('press Enter...')
    pl.plot(finished=True)

if __name__ == '__main__':
    main()
