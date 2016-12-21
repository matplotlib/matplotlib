# -*- noplot -*-
# Demo of using multiprocessing for generating data in one process and plotting
# in another.
# Written by Robert Cimrman

from __future__ import print_function
import time
from multiprocessing import Process, Pipe
import numpy as np
import matplotlib
# not all backends may allow safe plotting from multiple threads
# you can select a specific backend by uncommenting the line below
# and update the selected backend as needed
# matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while True:
            if not self.pipe.poll():
                break

            command = self.pipe.recv()

            if command is None:
                self.terminate()
                return False

            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')

        self.fig.canvas.draw_idle()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

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

    pl.plot(finished=True)

if __name__ == '__main__':
    # The default way to start a process on OSX ('fork')
    # does not work well with many gui frameworks on OSX
    # if you use Python 3.4 or later you can uncomment the
    # two lines below to change the default start to
    # forkserver.
    # import multiprocessing as mp
    # mp.set_start_method('forkserver')
    main()
