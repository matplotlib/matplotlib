# Demo of using multiprocessing for generating data in one process and plotting
# in another.
# Written by Robert Cimrman

from __future__ import print_function
import time
from multiprocessing import Process, Pipe
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def poll_draw(self):

        def call_back():
            while 1:
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

            self.fig.canvas.draw()
            return True

        return call_back()

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.poll_draw)
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
    raw_input('press Enter...')
#    input('press Enter...')    #Python3 
    pl.plot(finished=True)

if __name__ == '__main__':
    main()
